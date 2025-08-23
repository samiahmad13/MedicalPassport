from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from uuid import uuid4

from a2a.types import (
    AgentCard,
    AgentSkill,
    AgentCapabilities,
    Message,
    Role,
    Part,
    DataPart,
)
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.server.apps import A2AStarletteApplication


from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from .base_agent import BaseAgent, logger


# AgentCard
CARD = AgentCard(
    name="Summarizer Agent",
    version="1.0.0",
    description="Produces clinician-facing summary and risk flags from raw text and/or a FHIR-like bundle via MCP",
    url="http://localhost:41244/",
    capabilities=AgentCapabilities(),
    default_input_modes=["application/json"],
    default_output_modes=["application/json"],
    skills=[
        AgentSkill(
            id="run",
            name="run",
            description="Summarize and flag risks given clinical text and/or FHIR-like bundle",
            tags=["summary", "risk", "handoff", "llm"],
        )
    ],
)


# MCP client
class SummarizerAgent(BaseAgent):
    """Uses MCP risk_assessment tool (LLM) to produce a summary and list of risks."""

    def __init__(self, session: ClientSession):
        super().__init__(session, CARD)

    async def run(
        self,
        text: Optional[str] = None,
        bundle: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, list[str]]:
        self.ensure_skill("run")
        await self.ensure_tool("risk_assessment")

        payload: Dict[str, Any] = {}
        if text:
            payload["text"] = text
        if bundle:
            payload["bundle"] = bundle

        res = await self.call_tool_logged("risk_assessment", payload)
        out = self.extract_json("risk_assessment", res)
        summary = (out or {}).get("summary", "") or ""
        risks = (out or {}).get("risks", []) or []
        if not isinstance(risks, list):
            raise RuntimeError("MCP risk_assessment returned non-list risks")
        return summary, risks


# A2A Executor
class SummarizerAgentExecutor(AgentExecutor):
    """
    Accepts a Message with a DataPart:
      { "text": "...optional...", "bundle": { ...optional... } }

    Returns a Message with a DataPart:
      { "summary": "...", "risks": ["...", "..."] }
    """

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        logger.info("[A2A] %s skill=%s", CARD.name, "run")
        text, bundle = self._extract_payload(context)

        params = StdioServerParameters(
            command=sys.executable,
            args=[str(Path("mcp_server.py").resolve())],
            env=os.environ.copy(),
        )
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                agent = SummarizerAgent(session)
                summary, risks = await agent.run(text=text, bundle=bundle)

        message = Message(
            messageId=str(uuid4()),
            role=Role.agent,
            parts=[Part(root=DataPart(data={"summary": summary, "risks": risks}))],
        )
        await event_queue.enqueue_event(message)

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise Exception("cancel not supported")

    @staticmethod
    def _extract_payload(
        context: RequestContext,
    ) -> tuple[Optional[str], Optional[Dict[str, Any]]]:
        msg = getattr(context, "message", None)
        if msg and getattr(msg, "parts", None):
            for part in msg.parts:
                root = getattr(part, "root", None)
                # DataPart path
                if hasattr(root, "data") and isinstance(root.data, dict):
                    data = root.data
                    text = data.get("text")
                    bundle = data.get("bundle")
                    if text is not None and not isinstance(text, str):
                        raise RuntimeError("text must be a string if provided")
                    if bundle is not None and not isinstance(bundle, dict):
                        raise RuntimeError("bundle must be a JSON object if provided")
                    return text, bundle
                # TextPart with JSON body
                if hasattr(root, "text") and isinstance(root.text, str):
                    try:
                        data = json.loads(root.text)
                        text = data.get("text")
                        bundle = data.get("bundle")
                        if text is not None and not isinstance(text, str):
                            raise RuntimeError("text must be a string if provided")
                        if bundle is not None and not isinstance(bundle, dict):
                            raise RuntimeError(
                                "bundle must be a JSON object if provided"
                            )
                        return text, bundle
                    except Exception:
                        pass
        # both optional, at least one must be present for a decent summary
        raise RuntimeError(
            "Expect DataPart or JSON TextPart with 'text' and/or 'bundle' keys."
        )


# A2A Server Entrypoint
if __name__ == "__main__":
    import uvicorn

    host = os.getenv("A2A_HOST", "127.0.0.1")
    port = int(os.getenv("A2A_PORT", "41244"))

    public_card = CARD.model_copy(update={"url": f"http://{host}:{port}/"})

    handler = DefaultRequestHandler(
        agent_executor=SummarizerAgentExecutor(),
        task_store=InMemoryTaskStore(),
    )
    app_builder = A2AStarletteApplication(
        agent_card=public_card,
        http_handler=handler,
        extended_agent_card=None,
    )

    uvicorn.run(app_builder.build(), host=host, port=port)

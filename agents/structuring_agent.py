from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict
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
    name="Structuring Agent",
    version="1.0.0",
    description="Maps narrative clinical text into a minimal FHIR-like Bundle using MCP",
    url="http://localhost:41243/",
    capabilities=AgentCapabilities(),
    default_input_modes=["application/json"],
    default_output_modes=["application/json"],
    skills=[
        AgentSkill(
            id="run",
            name="run",
            description="Convert clinical narrative text to a minimal FHIR-like Bundle",
            tags=["structuring", "fhir", "llm"],
        )
    ],
)


# MCP client
class StructuringAgent(BaseAgent):
    """Uses MCP clinical_parse_to_fhir tool (LLM) to produce a minimal Bundle."""

    def __init__(self, session: ClientSession):
        super().__init__(session, CARD)

    async def run(
        self, text: str, patient_meta: Dict[str, Any] | None = None
    ) -> Dict[str, Any]:
        self.ensure_skill("run")
        await self.ensure_tool("clinical_parse_to_fhir")

        res = await self.call_tool_logged(
            "clinical_parse_to_fhir",
            {"text": text, "patient_meta": patient_meta or {}},
        )
        out = self.extract_json("clinical_parse_to_fhir", res)
        if not isinstance(out, dict) or "bundle" not in out:
            raise RuntimeError("MCP clinical_parse_to_fhir returned unexpected shape")
        return {"bundle": out["bundle"]}


# A2A Executor
class StructuringAgentExecutor(AgentExecutor):
    """
    Accepts a Message with a DataPart:
      { "text": "narrative clinical note", "patient_meta": { ... }? }

    Returns a Message with a DataPart:
      { "bundle": { "resourceType": "Bundle", ... } }
    """

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        logger.info("[A2A] %s skill=%s", CARD.name, "run")
        text, meta = self._extract_payload(context)

        params = StdioServerParameters(
            command=sys.executable,
            args=[str(Path("mcp_server.py").resolve())],
            env=os.environ.copy(),
        )
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                agent = StructuringAgent(session)
                result = await agent.run(text=text, patient_meta=meta)

        message = Message(
            messageId=str(uuid4()),
            role=Role.agent,
            parts=[Part(root=DataPart(data=result))],
        )
        await event_queue.enqueue_event(message)

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise Exception("cancel not supported")

    @staticmethod
    def _extract_payload(context: RequestContext) -> tuple[str, Dict[str, Any] | None]:
        msg = getattr(context, "message", None)
        if msg and getattr(msg, "parts", None):
            for part in msg.parts:
                root = getattr(part, "root", None)
                if hasattr(root, "data") and isinstance(root.data, dict):
                    data = root.data
                    text = data.get("text")
                    meta = data.get("patient_meta")
                    if isinstance(text, str):
                        return text, meta if isinstance(meta, dict) else None
                if hasattr(root, "text") and isinstance(root.text, str):
                    try:
                        data = json.loads(root.text)
                        text = data.get("text")
                        meta = data.get("patient_meta")
                        if isinstance(text, str):
                            return text, meta if isinstance(meta, dict) else None
                    except Exception:
                        pass
        raise RuntimeError(
            "Missing required inputs. Expect DataPart or JSON TextPart with key: text (and optional patient_meta)."
        )


# A2A Server Entrypoint
if __name__ == "__main__":
    import uvicorn

    host = os.getenv("A2A_HOST", "127.0.0.1")
    port = int(os.getenv("A2A_PORT", "41243"))

    public_card = CARD.model_copy(update={"url": f"http://{host}:{port}/"})

    handler = DefaultRequestHandler(
        agent_executor=StructuringAgentExecutor(),
        task_store=InMemoryTaskStore(),
    )
    app_builder = A2AStarletteApplication(
        agent_card=public_card,
        http_handler=handler,
        extended_agent_card=None,
    )

    uvicorn.run(app_builder.build(), host=host, port=port)

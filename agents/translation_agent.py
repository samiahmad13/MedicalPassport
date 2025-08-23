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
    name="Translation Agent",
    version="1.0.0",
    description="Detects source language and translates to a target locale using MCP LLM tool",
    url="http://localhost:41242/",
    capabilities=AgentCapabilities(),
    default_input_modes=["application/json"],
    default_output_modes=["application/json"],
    skills=[
        AgentSkill(
            id="run",
            name="run",
            description="Translate text to a target locale",
            tags=["translation", "language", "llm"],
        )
    ],
)


# MCP client
class TranslationAgent(BaseAgent):
    """Uses MCP translate_text tool (LLM) to translate and report detected source locale."""

    def __init__(self, session: ClientSession):
        super().__init__(session, CARD)

    async def run(self, text: str, target_locale: str) -> Dict[str, Any]:
        self.ensure_skill("run")
        await self.ensure_tool("translate_text")

        res = await self.call_tool_logged(
            "translate_text",
            {"text": text, "target_locale": target_locale},
        )
        out = self.extract_json("translate_text", res)
        # expected shape from MCP tool: {"text": "...", "source_locale": "xx"}
        translated = (out or {}).get("text", "")
        source_locale = (out or {}).get("source_locale", "und")
        return {"text": translated, "source_locale": source_locale}


# A2A Executor
class TranslationAgentExecutor(AgentExecutor):
    """
    Accepts a Message with a DataPart:
      { "text": "...", "target_locale": "en" }

    Returns a Message with a DataPart:
      { "text": "...translated...", "source_locale": "xx" }
    """

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        logger.info("[A2A] %s skill=%s", CARD.name, "run")
        # 1) Extract payload from incoming message
        text, target_locale = self._extract_payload(context)

        # 2) Create an MCP stdio session to mcp_server.py
        params = StdioServerParameters(
            command=sys.executable,
            args=[str(Path("mcp_server.py").resolve())],
            env=os.environ.copy(),
        )

        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                agent = TranslationAgent(session)
                result = await agent.run(text=text, target_locale=target_locale)

        # 3) Emit A2A reply as a Message with required messageId + DataPart
        message = Message(
            messageId=str(uuid4()),
            role=Role.agent,
            parts=[Part(root=DataPart(data=result))],
        )
        await event_queue.enqueue_event(message)

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise Exception("cancel not supported")

    @staticmethod
    def _extract_payload(context: RequestContext) -> tuple[str, str]:
        """
        Accept either:
          - DataPart with {"text": "...", "target_locale": "..."}
          - TextPart containing a JSON string with those keys
        """
        msg = getattr(context, "message", None)
        if msg and getattr(msg, "parts", None):
            for part in msg.parts:
                root = getattr(part, "root", None)
                if hasattr(root, "data") and isinstance(root.data, dict):
                    data = root.data
                    text = data.get("text")
                    target = data.get("target_locale")
                    if isinstance(text, str) and isinstance(target, str):
                        return text, target
                if hasattr(root, "text") and isinstance(root.text, str):
                    try:
                        data = json.loads(root.text)
                        text = data.get("text")
                        target = data.get("target_locale")
                        if isinstance(text, str) and isinstance(target, str):
                            return text, target
                    except Exception:
                        pass
        raise RuntimeError(
            "Missing required inputs. Expect DataPart or JSON TextPart with keys: text, target_locale."
        )


# A2A Server Entrypoint
if __name__ == "__main__":
    import uvicorn

    host = os.getenv("A2A_HOST", "127.0.0.1")
    port = int(os.getenv("A2A_PORT", "41242"))

    public_card = CARD.model_copy(update={"url": f"http://{host}:{port}/"})

    handler = DefaultRequestHandler(
        agent_executor=TranslationAgentExecutor(),
        task_store=InMemoryTaskStore(),
    )
    app_builder = A2AStarletteApplication(
        agent_card=public_card,
        http_handler=handler,
        extended_agent_card=None,
    )

    uvicorn.run(app_builder.build(), host=host, port=port)

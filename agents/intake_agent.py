from __future__ import annotations
from uuid import uuid4
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

from a2a.types import AgentCard, AgentSkill, AgentCapabilities, Message, Role, Part
from a2a.types import DataPart
from a2a.server.agent_execution import (
    AgentExecutor,
    RequestContext,
)
from a2a.server.events import EventQueue
from a2a.server.request_handlers import (
    DefaultRequestHandler,
)
from a2a.server.tasks import InMemoryTaskStore
from a2a.server.apps import A2AStarletteApplication

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from .base_agent import BaseAgent, logger


# AgentCard
CARD = AgentCard(
    name="Intake Agent",
    version="1.0.0",
    description="Performs OCR and language detection",
    url="http://localhost:41241/",
    capabilities=AgentCapabilities(),
    default_input_modes=["application/json"],
    default_output_modes=["application/json"],
    skills=[
        AgentSkill(
            id="run",
            name="run",
            description="Perform OCR and language detection",
            tags=["intake", "ocr", "language"],
        )
    ],
)


# MCP client
class IntakeAgent(BaseAgent):
    """Handles OCR and language detection via MCP tools."""

    def __init__(self, session: ClientSession):
        super().__init__(session, CARD)

    async def run(self, image_path: Path, locale_hint: str):
        self.ensure_skill("run")

        # OCR
        await self.ensure_tool("ocr_image")
        ocr_res = await self.call_tool_logged(
            "ocr_image", {"file_path": str(image_path), "locale_hint": locale_hint}
        )
        ocr = self.extract_json("ocr_image", ocr_res)
        text = ocr.get("text", "") or ""

        # Language detection
        await self.ensure_tool("detect_language")
        det_res = await self.call_tool_logged("detect_language", {"text": text})
        det = self.extract_json("detect_language", det_res)
        patient_lang = det.get("lang", "en")
        return text, patient_lang


# A2A Executor
class IntakeAgentExecutor(AgentExecutor):
    """
    A2A-compliant executor that:
      - Accepts a message with a DataPart containing {"file_path": "...", "locale_hint": "..."}.
      - Spins up an MCP stdio client to your mcp_server.py and calls the same IntakeAgent.run().
      - Emits a single Message with a DataPart containing {"text": ..., "patient_lang": ..., "metadata": {...}}.
    """

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        logger.info("[A2A] %s skill=%s", CARD.name, "run")
        # Extract payload from the incoming A2A message
        file_path, locale_hint = self._extract_payload(context)

        # Create a fresh MCP stdio session (simple & robust for MVP)
        params = StdioServerParameters(
            command=sys.executable,
            args=[str(Path("mcp_server.py").resolve())],
            env=os.environ.copy(),
        )

        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()

                intake = IntakeAgent(session)
                text, patient_lang = await intake.run(Path(file_path), locale_hint)

        result: Dict[str, Any] = {
            "text": text,
            "patient_lang": patient_lang,
            "metadata": {"source": file_path, "locale_hint": locale_hint},
        }

        message = Message(
            messageId=str(uuid4()),
            role=Role.agent,
            parts=[Part(root=DataPart(data=result))],
        )
        await event_queue.enqueue_event(message)

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise Exception("cancel not supported")

    # helpers
    @staticmethod
    def _extract_payload(context: RequestContext) -> tuple[str, str]:
        """
        Accept either:
          - DataPart with {"file_path": "...", "locale_hint": "..."}
          - TextPart containing a JSON string with the same keys
        """
        msg = getattr(context, "message", None)
        if msg and getattr(msg, "parts", None):
            for part in msg.parts:
                root = getattr(part, "root", None)
                if hasattr(root, "data") and isinstance(root.data, dict):
                    file_path = root.data.get("file_path")
                    locale_hint = root.data.get("locale_hint")
                    if file_path and locale_hint:
                        return str(file_path), str(locale_hint)
                if hasattr(root, "text") and isinstance(root.text, str):
                    try:
                        data = json.loads(root.text)
                        file_path = data.get("file_path")
                        locale_hint = data.get("locale_hint")
                        if file_path and locale_hint:
                            return str(file_path), str(locale_hint)
                    except Exception:
                        pass
        raise RuntimeError(
            "Missing required inputs. Expect DataPart or JSON TextPart with keys: file_path, locale_hint."
        )


# A2A Server Entrypoint
if __name__ == "__main__":
    import uvicorn

    host = os.getenv("A2A_HOST", "127.0.0.1")
    port = int(os.getenv("A2A_PORT", "41241"))

    public_card = CARD.model_copy(update={"url": f"http://{host}:{port}/"})

    handler = DefaultRequestHandler(
        agent_executor=IntakeAgentExecutor(),
        task_store=InMemoryTaskStore(),
    )
    app_builder = A2AStarletteApplication(
        agent_card=public_card,
        http_handler=handler,
        extended_agent_card=None,
    )

    uvicorn.run(app_builder.build(), host=host, port=port)

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
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

DEFAULT_CLINIC_FONT = Path("data/fonts/NotoSans-Regular.ttf").as_posix()
DEFAULT_PATIENT_FONT = Path("data/fonts/NotoNaskhArabic-Regular.ttf").as_posix()

CARD = AgentCard(
    name="Referral Packet Agent",
    version="1.0.0",
    description="Generates bilingual referral PDF and TXT via MCP pdf_generate",
    url="http://localhost:41245/",
    capabilities=AgentCapabilities(),
    default_input_modes=["application/json"],
    default_output_modes=["application/json"],
    skills=[
        AgentSkill(
            id="run",
            name="run",
            description="Create referral packet (PDF + TXT) from bundle, summaries, and risks",
            tags=["referral", "pdf", "handoff", "bilingual"],
        )
    ],
)


class ReferralAgent(BaseAgent):
    """Uses MCP pdf_generate to produce a PDF/TXT referral packet."""

    def __init__(
        self,
        session: ClientSession,
        clinic_font_path: Optional[str] = None,
        patient_font_path: Optional[str] = None,
    ):
        super().__init__(session, CARD)
        self.clinic_font_path = clinic_font_path or DEFAULT_CLINIC_FONT
        self.patient_font_path = patient_font_path or DEFAULT_PATIENT_FONT

    async def run(
        self,
        bundle: Dict[str, Any],
        summary_clinic: str,
        summary_patient: str,
        risks_clinic: List[str],
        risks_patient: List[str],
        out_dir: str = "data/outputs",
        title: str = "Medical Passport Referral",
        clinic_font_path: Optional[str] = None,
        patient_font_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        self.ensure_skill("run")
        await self.ensure_tool("pdf_generate")

        clinic_fp = clinic_font_path or self.clinic_font_path
        patient_fp = patient_font_path or self.patient_font_path

        payload = {
            "bundle": bundle,
            "summary_clinic": summary_clinic,
            "summary_patient": summary_patient,
            "risks_clinic": risks_clinic,
            "risks_patient": risks_patient,
            "out_dir": out_dir,
            "clinic_font_path": clinic_fp,
            "patient_font_path": patient_fp,
            "title": title,
        }
        res = await self.call_tool_logged("pdf_generate", payload)
        out = self.extract_json("pdf_generate", res)
        # Expected keys: pdf_path, txt_path, summary_clinic, summary_patient, risks_clinic, risks_patient
        return out


class ReferralAgentExecutor(AgentExecutor):
    """
    Accepts a Message with a DataPart:
      {
        "bundle": {...},
        "summary_clinic": "...",
        "summary_patient": "...",
        "risks_clinic": ["..."],
        "risks_patient": ["..."],
        "out_dir": "data/outputs",                (optional)
        "title": "Medical Passport Referral",     (optional)
        "clinic_font_path": "data/fonts/...",     (optional; REQUIRED by MCP if defaults absent)
        "patient_font_path": "data/fonts/..."     (optional; REQUIRED by MCP if defaults absent)
      }

    Returns a Message with a DataPart:
      {
        "pdf_path": ".../referral-YYYYMMDD-HHMMSS.pdf",
        "txt_path": ".../referral-YYYYMMDD-HHMMSS.txt",
        "summary_clinic": "...",
        "summary_patient": "...",
        "risks_clinic": [...],
        "risks_patient": [...]
      }
    """

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        logger.info("[A2A] %s skill=%s", CARD.name, "run")
        (
            bundle,
            summary_clinic,
            summary_patient,
            risks_clinic,
            risks_patient,
            out_dir,
            title,
            clinic_font_path,
            patient_font_path,
        ) = self._extract_payload(context)

        params = StdioServerParameters(
            command=sys.executable,
            args=[str(Path("mcp_server.py").resolve())],
            env=os.environ.copy(),
        )
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                agent = ReferralAgent(session)
                result = await agent.run(
                    bundle=bundle,
                    summary_clinic=summary_clinic,
                    summary_patient=summary_patient,
                    risks_clinic=risks_clinic,
                    risks_patient=risks_patient,
                    out_dir=out_dir,
                    title=title,
                    clinic_font_path=clinic_font_path,
                    patient_font_path=patient_font_path,
                )

        message = Message(
            messageId=str(uuid4()),
            role=Role.agent,
            parts=[Part(root=DataPart(data=result))],
        )
        await event_queue.enqueue_event(message)

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise Exception("cancel not supported")

    @staticmethod
    def _extract_payload(context: RequestContext):
        msg = getattr(context, "message", None)
        if msg and getattr(msg, "parts", None):
            for part in msg.parts:
                root = getattr(part, "root", None)

                def parse_dict(d: Dict[str, Any]):
                    bundle = d.get("bundle")
                    summary_clinic = d.get("summary_clinic")
                    summary_patient = d.get("summary_patient")
                    risks_clinic = d.get("risks_clinic")
                    risks_patient = d.get("risks_patient")
                    out_dir = d.get("out_dir", "data/outputs")
                    title = d.get("title", "Medical Passport Referral")
                    clinic_font_path = d.get("clinic_font_path")
                    patient_font_path = d.get("patient_font_path")

                    if not isinstance(bundle, dict):
                        raise RuntimeError("bundle (dict) is required")
                    if not isinstance(summary_clinic, str) or not isinstance(
                        summary_patient, str
                    ):
                        raise RuntimeError(
                            "summary_clinic and summary_patient (str) are required"
                        )
                    if not isinstance(risks_clinic, list) or not isinstance(
                        risks_patient, list
                    ):
                        raise RuntimeError(
                            "risks_clinic and risks_patient (list[str]) are required"
                        )

                    return (
                        bundle,
                        summary_clinic,
                        summary_patient,
                        risks_clinic,
                        risks_patient,
                        out_dir,
                        title,
                        clinic_font_path,
                        patient_font_path,
                    )

                # DataPart path
                if hasattr(root, "data") and isinstance(root.data, dict):
                    return parse_dict(root.data)

                # TextPart path(JSON body)
                if hasattr(root, "text") and isinstance(root.text, str):
                    try:
                        data = json.loads(root.text)
                        if isinstance(data, dict):
                            return parse_dict(data)
                    except Exception:
                        pass

        raise RuntimeError(
            "Expect DataPart or JSON TextPart with keys: bundle, summary_clinic, summary_patient, risks_clinic, risks_patient"
        )


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("A2A_HOST", "127.0.0.1")
    port = int(os.getenv("A2A_PORT", "41245"))

    public_card = CARD.model_copy(update={"url": f"http://{host}:{port}/"})

    handler = DefaultRequestHandler(
        agent_executor=ReferralAgentExecutor(),
        task_store=InMemoryTaskStore(),
    )
    app_builder = A2AStarletteApplication(
        agent_card=public_card,
        http_handler=handler,
        extended_agent_card=None,
    )

    uvicorn.run(app_builder.build(), host=host, port=port)

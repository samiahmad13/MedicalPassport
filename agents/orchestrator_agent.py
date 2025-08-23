from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple
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

import httpx
from a2a.client.card_resolver import A2ACardResolver
from a2a.client.legacy import A2AClient
from a2a.types import SendMessageRequest, MessageSendParams


# AgentCard
CARD = AgentCard(
    name="Orchestrator Agent",
    version="1.0.0",
    description="Coordinates Intake→Translation→Structuring→Summarizer→Referral via A2A, on top of MCP tools",
    url="http://localhost:41246/",
    capabilities=AgentCapabilities(),
    default_input_modes=["application/json"],
    default_output_modes=["application/json"],
    skills=[
        AgentSkill(
            id="orchestrate",
            name="orchestrate",
            description="Run end-to-end MedicalPassport pipeline and return PDF/TXT paths",
            tags=["orchestrator", "workflow", "handoff"],
        )
    ],
)


# A2A call utils
async def _resolve_card(http: httpx.AsyncClient, base_url: str):
    resolver = A2ACardResolver(httpx_client=http, base_url=base_url)
    return await resolver.get_agent_card()


FONT_MAP = {
    "ar": "data/fonts/NotoNaskhArabic-Regular.ttf",
    "es": "data/fonts/NotoSans-Regular.ttf",
    "en": "data/fonts/NotoSans-Regular.ttf",
    # need to add way more
}


def resolve_font(lang_code: str) -> str:
    return FONT_MAP.get(lang_code, "data/fonts/NotoSans-Regular.ttf")


async def _send_datapart(
    http: httpx.AsyncClient,
    base_url: str,
    payload: Dict[str, Any],
    timeout_sec: float = 180.0,
) -> Dict[str, Any] | None:
    """
    Send a non-streaming A2A message with one DataPart, return the first DataPart from result.
    Accepts both {'type': 'data', ...} and {'kind': 'data', ...} shapes.
    """
    card = await _resolve_card(http, base_url)
    client = A2AClient(agent_card=card, httpx_client=http)

    req = SendMessageRequest(
        id=str(uuid4()),
        params=MessageSendParams(
            message={
                "role": "user",
                "messageId": uuid4().hex,
                "parts": [{"kind": "data", "data": payload}],
            }
        ),
    )
    resp = await client.send_message(req)
    dumped = resp.model_dump(mode="json", exclude_none=True)

    if "error" in dumped and dumped["error"]:
        return {"_error": dumped["error"]}

    parts = (
        dumped.get("result", {}).get("parts")
        or dumped.get("result", {}).get("message", {}).get("parts")
        or []
    )

    # Accept both shapes: {"type": "data", "data": {...}} or {"kind": "data", "data": {...}}
    for p in parts:
        if ("data" in p) and (p.get("type") == "data" or p.get("kind") == "data"):
            return p["data"]

    # Fallback for debugging if no data part found
    return {"_raw": dumped}


# Executor
class OrchestratorAgentExecutor(AgentExecutor):
    """
    Input DataPart:
      {
        "image_path": "data/samples/note_ar.jpg",
        "locale_hint": "ara",
        "patient_lang_target": "en"    # optional, default "en"
      }

    Output DataPart:
      {
        "pdf_path": "...",
        "txt_path": "...",
        "patient_lang": "ar",
        "bundle": {...},
        "summary_en": "...",
        "risks_en": [...]
      }
    """

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        image_path, locale_hint, patient_lang_target = self._extract_payload(context)
        # Base URLs (override via env if ports changed)
        intake_url = os.getenv("INTAKE_URL", "http://127.0.0.1:41241")
        translate_url = os.getenv("TRANSLATE_URL", "http://127.0.0.1:41242")
        structuring_url = os.getenv("STRUCTURE_URL", "http://127.0.0.1:41243")
        summarizer_url = os.getenv("SUMMARIZER_URL", "http://127.0.0.1:41244")
        referral_url = os.getenv("REFERRAL_URL", "http://127.0.0.1:41245")

        async with httpx.AsyncClient(timeout=180.0) as http:

            # 1) Intake
            intake_out = await _send_datapart(
                http,
                intake_url,
                {"file_path": image_path, "locale_hint": locale_hint},
            )
            if isinstance(intake_out, dict) and "_error" in intake_out:
                raise RuntimeError(f"Intake error: {intake_out['_error']}")

            if not isinstance(intake_out, dict) or "text" not in intake_out:
                raise RuntimeError(f"Intake returned unexpected: {intake_out}")
            raw_text = intake_out.get("text", "")
            patient_lang = intake_out.get("patient_lang", "und")

            # 2) Translate to working language (default "en")
            tr_en = await _send_datapart(
                http,
                translate_url,
                {"text": raw_text, "target_locale": patient_lang_target},
            )
            if isinstance(tr_en, dict) and "_error" in tr_en:
                raise RuntimeError(f"Intake error: {tr_en['_error']}")

            if not isinstance(tr_en, dict) or "text" not in tr_en:
                raise RuntimeError(f"Translation returned unexpected: {tr_en}")
            text_en = tr_en["text"]

            # 3) Structure to FHIR-like Bundle
            fhir_out = await _send_datapart(
                http,
                structuring_url,
                {"text": text_en},
            )
            if isinstance(fhir_out, dict) and "_error" in fhir_out:
                raise RuntimeError(f"Intake error: {fhir_out['_error']}")

            if not isinstance(fhir_out, dict) or "bundle" not in fhir_out:
                raise RuntimeError(f"Structuring returned unexpected: {fhir_out}")
            bundle = fhir_out["bundle"]
            # 4) Summarize & risk assess (default clinic language = en)
            sum_out = await _send_datapart(
                http,
                summarizer_url,
                {"text": text_en, "bundle": bundle},
            )
            if isinstance(sum_out, dict) and "_error" in sum_out:
                raise RuntimeError(f"Intake error: {sum_out['_error']}")

            if (
                not isinstance(sum_out, dict)
                or "summary" not in sum_out
                or "risks" not in sum_out
            ):
                raise RuntimeError(f"Summarizer returned unexpected: {sum_out}")
            summary_en = sum_out["summary"]
            risks_en: List[str] = sum_out["risks"]

            # 5) Patient-facing translation (if needed)
            if patient_lang and patient_lang != patient_lang_target:
                tr_summary_patient = await _send_datapart(
                    http,
                    translate_url,
                    {"text": summary_en, "target_locale": patient_lang},
                )
                if (
                    isinstance(tr_summary_patient, dict)
                    and "_error" in tr_summary_patient
                ):
                    raise RuntimeError(f"Intake error: { tr_summary_patient['_error']}")

                summary_patient = tr_summary_patient.get("text", summary_en)

                risks_patient: List[str] = []
                for r in risks_en:
                    tr = await _send_datapart(
                        http,
                        translate_url,
                        {"text": r, "target_locale": patient_lang},
                    )
                    if isinstance(tr, dict) and "_error" in tr:
                        raise RuntimeError(f"Intake error: { tr['_error']}")
                    risks_patient.append(tr.get("text", r))
            else:
                summary_patient = summary_en
                risks_patient = risks_en

            # 6) Referral packet (PDF/TXT)
            referral_in = {
                "bundle": bundle,
                "summary_clinic": summary_en,
                "summary_patient": summary_patient,
                "risks_clinic": risks_en,
                "risks_patient": risks_patient,
                "out_dir": "data/outputs",
                "title": "Medical Passport Referral",
                "clinic_font_path": resolve_font("en"),
                "patient_font_path": resolve_font(patient_lang),
            }
            ref_out = await _send_datapart(http, referral_url, referral_in)

            if isinstance(ref_out, dict) and "_error" in ref_out:
                raise RuntimeError(f"Intake error: {ref_out['_error']}")

            if (
                not isinstance(ref_out, dict)
                or "pdf_path" not in ref_out
                or "txt_path" not in ref_out
            ):
                raise RuntimeError(f"Referral returned unexpected: {ref_out}")

        # Emit final message
        clinical_msg = f"PDF → {ref_out['pdf_path']}  |  TXT → {ref_out['txt_path']}  |  Patient language: {patient_lang}"

        final_data = {
            "pdf_path": ref_out["pdf_path"],
            "txt_path": ref_out["txt_path"],
            "patient_lang": patient_lang,
            "bundle": bundle,
            "summary_en": summary_en,
            "risks_en": risks_en,
            "final_message": clinical_msg,
        }

        message = Message(
            messageId=str(uuid4()),
            role=Role.agent,
            parts=[Part(root=DataPart(data=final_data))],
        )
        await event_queue.enqueue_event(message)

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise Exception("cancel not supported")

    # helpers
    @staticmethod
    def _extract_payload(context: RequestContext) -> Tuple[str, str, str]:
        msg = getattr(context, "message", None)
        if msg and getattr(msg, "parts", None):
            for part in msg.parts:
                root = getattr(part, "root", None)
                if hasattr(root, "data") and isinstance(root.data, dict):
                    d = root.data
                    image_path = d.get("image_path")
                    locale_hint = d.get("locale_hint")
                    target = d.get("patient_lang_target", "en")
                    if not isinstance(image_path, str) or not isinstance(
                        locale_hint, str
                    ):
                        raise RuntimeError(
                            "image_path (str) and locale_hint (str) are required"
                        )
                    if not isinstance(target, str):
                        raise RuntimeError(
                            "patient_lang_target must be a string if provided"
                        )
                    return image_path, locale_hint, target
                if hasattr(root, "text") and isinstance(root.text, str):
                    try:
                        d = json.loads(root.text)
                        image_path = d.get("image_path")
                        locale_hint = d.get("locale_hint")
                        target = d.get("patient_lang_target", "en")
                        if isinstance(image_path, str) and isinstance(locale_hint, str):
                            return (
                                image_path,
                                locale_hint,
                                target if isinstance(target, str) else "en",
                            )
                    except Exception:
                        pass
        raise RuntimeError(
            "Expect DataPart or JSON TextPart with keys: image_path, locale_hint, (optional) patient_lang_target"
        )


# Server Entrypoint
if __name__ == "__main__":
    import uvicorn

    host = os.getenv("A2A_HOST", "127.0.0.1")
    port = int(os.getenv("A2A_PORT", "41246"))

    public_card = CARD.model_copy(update={"url": f"http://{host}:{port}/"})
    handler = DefaultRequestHandler(
        agent_executor=OrchestratorAgentExecutor(),
        task_store=InMemoryTaskStore(),
    )
    app_builder = A2AStarletteApplication(
        agent_card=public_card,
        http_handler=handler,
        extended_agent_card=None,
    )
    uvicorn.run(app_builder.build(), host=host, port=port)

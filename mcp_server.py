from __future__ import annotations
import os, json, textwrap, datetime
from typing import Optional, Dict, Any
from mcp.server.fastmcp import FastMCP
from PIL import Image
import pytesseract
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from dotenv import load_dotenv
import openai
from langdetect import detect, detect_langs, DetectorFactory
import arabic_reshaper
from bidi.algorithm import get_display
import unicodedata, re

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "o4-mini")

if OPENAI_API_KEY:
    openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
else:
    openai_client = None

mcp = FastMCP("MedicalPassport-MCP")
DetectorFactory.seed = 0  # stabilize langdetect


# llm helpers
def _require_llm():
    if openai_client is None:
        raise RuntimeError("OPENAI_API_KEY is not set; this tool requires an LLM.")


def _llm_complete(system: str, user: str) -> str:
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    resp = openai_client.chat.completions.create(model=OPENAI_MODEL, messages=msgs)
    return resp.choices[0].message.content.strip()


# OCR Tool
@mcp.tool(
    name="ocr_image",
    description=(
        "Run OCR on an image file and return extracted text and metadata. "
        "STRICT: 'locale_hint' is REQUIRED and is passed directly to Tesseract (e.g., 'eng', 'ara')."
    ),
)
def ocr_image(file_path: str, locale_hint: Optional[str] = None) -> Dict[str, Any]:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"file_path not found: {file_path}")
    if not locale_hint or not isinstance(locale_hint, str) or not locale_hint.strip():
        raise RuntimeError(
            "locale_hint is required and must be a valid Tesseract language code (e.g., 'eng', 'ara')."
        )
    img = Image.open(file_path)
    used_lang = locale_hint.strip()
    text = pytesseract.image_to_string(img, lang=used_lang)
    return {
        "text": text,
        "metadata": {"source": file_path, "used_lang": used_lang},
    }


# Language Detection
@mcp.tool(name="detect_language", description="Detect language of text.")
def detect_language(text: str) -> Dict[str, Any]:
    text = (text or "").strip()
    if not text:
        return {"lang": "und", "confidence": 0.0, "alternates": []}
    try:
        candidates = detect_langs(text)
        alts = [
            {"lang": c.lang, "prob": float(getattr(c, "prob", 0.0))} for c in candidates
        ]
        top = (
            max(alts, key=lambda c: c["prob"]) if alts else {"lang": "und", "prob": 0.0}
        )
        return {
            "lang": top["lang"],
            "confidence": round(top["prob"], 3),
            "alternates": alts[:5],
        }
    except Exception:
        return {"lang": "und", "confidence": 0.0, "alternates": []}


# Translation
@mcp.tool(
    name="translate_text",
    description="Translate text to target locale using the LLM. Returns ONLY the translated text plus detected source locale.",
)
def translate_text(text: str, target_locale: str) -> Dict[str, Any]:
    _require_llm()
    system = (
        "You are a precise clinical translator. Translate the user's text to the requested target language. "
        "Preserve medication names and clinical terminology. Return ONLY the translated text."
    )
    user = f"Target language: {target_locale}\n\n---\n{text}"
    translated = _llm_complete(system, user)
    src = detect(text) if text.strip() else "und"
    return {"text": translated, "source_locale": src}


# Parse Clinical Note (FHIR-like JSON) - still need to add alignment with real coding systems
@mcp.tool(
    name="clinical_parse_to_fhir",
    description="Parse narrative clinical text into a minimal FHIR-like JSON Bundle using the LLM. Returns strict JSON only.",
)
def clinical_parse_to_fhir(
    text: str, patient_meta: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    _require_llm()
    system = (
        "You are an expert clinical coder. Convert the user's narrative clinical note into a minimal FHIR-like JSON Bundle. "
        "Respond with ONLY strict JSON, no code fences, no commentary. "
        "Follow this shape:\n"
        "{\n"
        '  "resourceType": "Bundle",\n'
        '  "type": "collection",\n'
        '  "entry": [\n'
        '    {"resource": {"resourceType":"Condition","code":{"text":""}}},\n'
        '    {"resource": {"resourceType":"MedicationStatement","medicationCodeableConcept":{"text":""}}},\n'
        '    {"resource": {"resourceType":"Procedure","code":{"text":""}}},\n'
        '    {"resource": {"resourceType":"Observation","code":{"text":""},"valueString":""}}\n'
        "  ]\n"
        "}\n"
        "Include only items that appear in the text. Use free-text codes (no coding systems)."
    )
    bundle_text = _llm_complete(system, text)
    try:
        bundle = json.loads(bundle_text)
    except Exception as e:
        raise RuntimeError(f"LLM did not return valid JSON Bundle: {e}")
    return {"bundle": bundle}


# Risk Assessment (llm)
@mcp.tool(
    name="risk_assessment",
    description="Summarize history and flag key risks from text or a FHIR-like bundle using the LLM. Returns plain text + a list of risks.",
)
def risk_assessment(
    text: Optional[str] = None, bundle: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    _require_llm()
    system = (
        "You are a clinical summarizer. Produce a 6-10 sentence summary for a handoff, "
        "AND list key risks as short bullet points (max 6). If both raw text and a FHIR-like bundle are present, rely on both. "
        "Return plain text; use lines starting with '- ' for risks."
    )
    user = f"TEXT:\n{(text or '').strip()}\n\nBUNDLE:\n{json.dumps(bundle, ensure_ascii=False, indent=2) if bundle else ''}"
    result = _llm_complete(system, user)

    lines = [ln.rstrip() for ln in result.splitlines()]
    risks = [ln.strip()[2:].strip() for ln in lines if ln.strip().startswith("- ")]
    if risks:
        first_idx = next(i for i, ln in enumerate(lines) if ln.strip().startswith("- "))
        summary = "\n".join(lines[:first_idx]).strip() or result.strip()
    else:
        summary = result.strip()

    return {"summary": summary, "risks": risks}


# PDF Generator
@mcp.tool(
    name="pdf_generate",
    description=(
        "Generate a bilingual referral PDF + plain-text from a bundle, two summaries, and two risk lists. "
        "Requires TrueType fonts for clinic_font_path and patient_font_path."
    ),
)
def pdf_generate(
    bundle: Dict[str, Any],
    summary_clinic: str,
    summary_patient: str,
    risks_clinic: list[str],
    risks_patient: list[str],
    out_dir: str = "data/outputs",
    clinic_font_path: str | None = None,
    patient_font_path: str | None = None,
    title: str = "Medical Passport Referral",
) -> Dict[str, Any]:

    if not clinic_font_path or not patient_font_path:
        raise RuntimeError(
            "pdf_generate requires clinic_font_path and patient_font_path (TTF)."
        )

    clinic_font_name = "ClinicFont"
    patient_font_name = "PatientFont"
    pdfmetrics.registerFont(TTFont(clinic_font_name, clinic_font_path))
    pdfmetrics.registerFont(TTFont(patient_font_name, patient_font_path))

    def _is_rtl(txt: str) -> bool:
        return any(
            (0x0590 <= ord(ch) <= 0x08FF) or (0xFB1D <= ord(ch) <= 0xFEFF)
            for ch in txt or ""
        )

    def _shape_for_pdf(txt: str) -> str:
        if not txt:
            return txt
        return get_display(arabic_reshaper.reshape(txt)) if _is_rtl(txt) else txt

    def _strip_risk_headings(text: str) -> str:
        """
        Remove lines that are just 'Key risks' headings in many languages.
        Robust to case, extra spaces, ASCII/fullwidth colons, and diacritics.
        """
        if not text:
            return text

        def _norm(s: str) -> str:
            s = unicodedata.normalize("NFKC", s or "").lower()
            s = s.strip().strip(":").strip("：").strip()
            s = re.sub(r"\s+", " ", s)
            s = "".join(
                ch
                for ch in unicodedata.normalize("NFKD", s)
                if not unicodedata.combining(ch)
            )
            return s

        # Source phrases
        kill_exact_raw = {
            # English
            "key risks",
            "risks",
            "main risks",
            "risk factors",
            # Spanish
            "riesgos clave",
            "riesgos",
            "principales riesgos",
            "factores de riesgo",
            # French
            "risques clés",
            "risques",
            "principaux risques",
            "facteurs de risque",
            # Portuguese
            "riscos principais",
            "riscos chave",
            "riscos",
            "fatores de risco",
            # Italian
            "rischi chiave",
            "rischi principali",
            "rischi",
            "fattori di rischio",
            # German
            "zentrale risiken",
            "wichtige risiken",
            "haupt­risiken",
            "risiken",
            "risikofaktoren",
            # Turkish
            "ana riskler",
            "önemli riskler",
            "riskler",
            "risk faktörleri",
            # Russian
            "ключевые риски",
            "основные риски",
            "риски",
            "факторы риска",
            # Arabic
            "المخاطر الرئيسية",
            "المخاطر",
            # Farsi
            "ریسک‌های کلیدی",
            "ریسک‌های اصلی",
            "ریسک‌ها",
            "عوامل خطر",
            # Urdu
            "اہم خطرات",
            "بنیادی خطرات",
            "خطرات",
            "خطر کے عوامل",
            # Hindi
            "मुख्य जोखिम",
            "प्रमुख जोखिम",
            "जोखिम",
            "जोखिम कारक",
            # Bengali
            "মূল ঝুঁকি",
            "প্রধান ঝুঁকি",
            "ঝুঁকি",
            "ঝুঁকির কারণ",
            # Chinese (Simplified)
            "关键风险",
            "主要风险",
            "风险",
            "风险因素",
            # Chinese (Traditional)
            "關鍵風險",
            "主要風險",
            "風險",
            "風險因素",
            # Indonesian / Malay
            "risiko utama",
            "risiko kunci",
            "risiko",
            "faktor risiko",
            # Swahili
            "hatari kuu",
            "hatari muhimu",
            "hatari",
            "vichocheo vya hatari",
            # Filipino
            "mahahalagang panganib",
            "pangunahing panganib",
            "mga panganib",
            "mga salik ng panganib",
        }

        kill_exact = {_norm(p) for p in kill_exact_raw}

        out = []
        for raw in text.splitlines():
            norm = _norm(raw)
            if norm in kill_exact:
                continue
            out.append(raw)
        return "\n".join(out).strip()

    summary_clinic_clean = _strip_risk_headings(summary_clinic)
    summary_patient_clean = _strip_risk_headings(summary_patient)

    summary_clinic_pdf = _shape_for_pdf(summary_clinic_clean)
    summary_patient_pdf = _shape_for_pdf(summary_patient_clean)
    risks_clinic_pdf = [_shape_for_pdf(x) for x in (risks_clinic or [])]
    risks_patient_pdf = [_shape_for_pdf(x) for x in (risks_patient or [])]

    # outputs
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    pdf_path = os.path.join(out_dir, f"referral-{ts}.pdf")
    txt_path = os.path.join(out_dir, f"referral-{ts}.txt")

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("=== Clinical Summary ===\n")
        f.write(summary_clinic_clean + "\n\n")
        if risks_clinic:
            f.write("=== Key Risks ===\n")
            for r in risks_clinic:
                f.write(f"- {r}\n")
            f.write("\n")

        f.write("=== Patient Summary ===\n")
        f.write(summary_patient_clean + "\n\n")
        if risks_patient:
            f.write("=== Key Risks (Patient) ===\n")
            for r in risks_patient:
                f.write(f"- {r}\n")
            f.write("\n")

        f.write("=== Structured Clinical Data ===\n")
        for line in _bundle_to_bullets(bundle):
            f.write(f"- {line}\n")
        f.write("\n=== Bundle (RAW JSON) ===\n")
        f.write(json.dumps(bundle, ensure_ascii=False, indent=2))
        f.write("\n")

    # pdf
    width, height = A4
    c = canvas.Canvas(pdf_path, pagesize=A4)

    # layout constants
    MARGIN_X = 50
    TOP_START = height - 60
    BOTTOM_MARGIN = 60
    TITLE_SIZE = 16
    H1_SIZE = 12
    BODY_SIZE = 11
    LINE_HEIGHT = 14
    RULE_WIDTH = 1.0
    RULE_GAP_BEFORE = 10
    RULE_GAP_AFTER = 16
    HEADING_GAP = 8
    SECTION_GAP = 22
    BULLET_INDENT = 14

    y = TOP_START

    def ensure_space(min_needed=LINE_HEIGHT):
        nonlocal y
        if y < (BOTTOM_MARGIN + min_needed):
            c.showPage()
            draw_header()

    def draw_rule():
        nonlocal y
        y -= RULE_GAP_BEFORE
        c.setLineWidth(RULE_WIDTH)
        c.line(MARGIN_X, y, width - MARGIN_X, y)
        y -= RULE_GAP_AFTER

    def draw_header():
        nonlocal y
        y = TOP_START
        c.setFont(clinic_font_name, TITLE_SIZE)
        c.drawCentredString(width / 2.0, y, title)
        y -= 16
        c.setFont(clinic_font_name, 10)
        c.drawCentredString(
            width / 2.0,
            y,
            f"Generated {datetime.datetime.utcnow().strftime('%Y-%m-%d')}",
        )
        y -= 8
        draw_rule()

    def draw_heading(txt: str, font: str):
        nonlocal y
        if "Key Risks" in txt:
            y -= SECTION_GAP
        ensure_space(H1_SIZE + HEADING_GAP + LINE_HEIGHT)
        c.setFont(font, H1_SIZE)
        c.drawString(MARGIN_X, y, txt)
        y -= H1_SIZE + HEADING_GAP

    def draw_paragraph(text: str, font: str, width_chars=100):
        nonlocal y
        ensure_space(LINE_HEIGHT)
        c.setFont(font, BODY_SIZE)
        for line in text.splitlines():
            segs = textwrap.wrap(
                line,
                width=width_chars,
                break_long_words=False,
                replace_whitespace=False,
            ) or [""]
            for seg in segs:
                c.drawString(MARGIN_X, y, seg)
                y -= LINE_HEIGHT
                ensure_space(LINE_HEIGHT)
        y -= 4

    def draw_bullets(items: list[str], font: str, width_chars=96):
        nonlocal y
        c.setFont(font, BODY_SIZE)
        for item in items:
            segs = textwrap.wrap(
                item,
                width=width_chars,
                break_long_words=False,
                replace_whitespace=False,
            ) or [""]
            ensure_space(LINE_HEIGHT)
            c.drawString(MARGIN_X, y, "• " + segs[0])
            y -= LINE_HEIGHT
            for cont in segs[1:]:
                ensure_space(LINE_HEIGHT)
                c.drawString(MARGIN_X + BULLET_INDENT, y, cont)
                y -= LINE_HEIGHT
            y -= 4

    # title
    draw_header()

    # Clinical Summary + Clinic Risks
    draw_heading("Clinical Summary", clinic_font_name)
    draw_paragraph(summary_clinic_pdf, clinic_font_name)
    if risks_clinic_pdf:
        draw_heading("Key Risks", clinic_font_name)
        draw_bullets(risks_clinic_pdf, clinic_font_name)
    draw_rule()

    # Patient Summary + Patient Risks
    draw_heading("Patient Summary", patient_font_name)
    draw_paragraph(summary_patient_pdf, patient_font_name)
    if risks_patient_pdf:
        draw_heading("Key Risks", patient_font_name)
        draw_bullets(risks_patient_pdf, patient_font_name)
    draw_rule()

    # Structured Clinical Data
    bullets = _bundle_to_bullets(bundle)
    if bullets:
        draw_heading("Structured Clinical Data", clinic_font_name)
        draw_bullets(bullets, clinic_font_name)

    c.save()
    return {
        "pdf_path": pdf_path,
        "txt_path": txt_path,
        "summary_clinic": summary_clinic_clean,
        "summary_patient": summary_patient_clean,
        "risks_clinic": risks_clinic,
        "risks_patient": risks_patient,
    }


def _bundle_to_bullets(bundle: Dict[str, Any]) -> list[str]:
    bullets: list[str] = []
    for e in bundle.get("entry", []):
        r = e.get("resource", {})
        try:
            rt = r.get("resourceType")
            if rt == "Condition":
                txt = (r.get("code", {}) or {}).get("text", "")
                if txt:
                    bullets.append(txt)
            elif rt == "MedicationStatement":
                med = (r.get("medicationCodeableConcept", {}) or {}).get("text", "")
                if med:
                    bullets.append(med)
            elif rt == "Procedure":
                txt = (r.get("code", {}) or {}).get("text", "")
                if txt:
                    bullets.append(txt)
            elif rt == "Observation":
                code = (r.get("code", {}) or {}).get("text", "")
                val = r.get("valueString", "")
                if code or val:
                    bullets.append(f"{code}" + (f" — {val}" if val else ""))
        except Exception:
            continue
    return bullets


if __name__ == "__main__":
    transport = os.getenv("MCP_TRANSPORT", "stdio").strip().lower()
    if transport in ("streamable-http", "http"):
        host = os.getenv("MCP_HOST", "127.0.0.1")
        port = int(os.getenv("MCP_PORT", "8000"))
        path = os.getenv("MCP_PATH", "/mcp")
        mcp.settings.host = host
        mcp.settings.port = port
        mcp.settings.streamable_http_path = path
        mcp.run(transport="streamable-http")
    elif transport == "sse":
        host = os.getenv("MCP_HOST", "127.0.0.1")
        port = int(os.getenv("MCP_PORT", "8000"))
        mcp.settings.host = host
        mcp.settings.port = port
        mcp.run(transport="sse")
    else:
        mcp.run()
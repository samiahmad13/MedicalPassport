from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
import sys
import time
from pathlib import Path
from uuid import uuid4
from typing import Dict, Any, List, Tuple
import functools
import httpx


from a2a.client.card_resolver import A2ACardResolver
from a2a.client.legacy import A2AClient
from a2a.types import SendMessageRequest, MessageSendParams


AGENTS: List[Tuple[str, int]] = [
    ("agents.intake_agent", 41241),
    ("agents.translation_agent", 41242),
    ("agents.structuring_agent", 41243),
    ("agents.summarizer_agent", 41244),
    ("agents.referral_agent", 41245),
    ("agents.orchestrator_agent", 41246),
]
CARD_PATH = "/.well-known/agent-card.json"


# health check
async def wait_ready(base_url: str, timeout: float = 120.0) -> None:
    start = time.time()
    async with httpx.AsyncClient(timeout=5.0) as http:
        while True:
            try:
                r = await http.get(base_url + CARD_PATH)
                if r.status_code == 200:
                    return
            except Exception:
                pass
            if time.time() - start > timeout:
                raise TimeoutError(f"Timed out waiting for {base_url}{CARD_PATH}")
            await asyncio.sleep(0.3)


# launch / cleanup
async def launch_all() -> list[asyncio.subprocess.Process]:
    procs: list[asyncio.subprocess.Process] = []

    async def pipe_stderr(mod: str, proc: asyncio.subprocess.Process):
        if proc.stderr is None:
            return
        while True:
            line = await proc.stderr.readline()
            if not line:
                break
            sys.stderr.write(f"[{mod}] {line.decode(errors='ignore')}")
            sys.stderr.flush()

    try:
        for mod, port in AGENTS:
            env = os.environ.copy()
            env["A2A_HOST"] = env.get("A2A_HOST", "127.0.0.1")
            env["A2A_PORT"] = str(port)

            print(f"[launch] starting {mod} on :{port}")
            proc = await asyncio.create_subprocess_exec(
                sys.executable,
                "-m",
                mod,
                env=env,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.PIPE,
            )
            procs.append(proc)

            asyncio.create_task(pipe_stderr(mod, proc))

            await wait_ready(f"http://127.0.0.1:{port}")
            print(f"[launch] {mod} ready on :{port}")
        print("[launch] all agents ready")
        return procs
    except Exception:
        for p in procs:
            if p.returncode is None:
                try:
                    p.send_signal(signal.SIGINT)
                    await asyncio.wait_for(p.wait(), timeout=5)
                except Exception:
                    p.kill()
        raise


async def cleanup(procs: List[asyncio.subprocess.Process]) -> None:
    for p in procs:
        if p.returncode is None:
            try:
                p.send_signal(signal.SIGINT)
                await asyncio.wait_for(p.wait(), timeout=5)
            except Exception:
                p.kill()


# A2A helper (non-streaming)
async def send_to_orchestrator(
    image_path: Path, locale_hint: str, target_lang: str = "en"
) -> Dict[str, Any]:
    orch_url = os.environ.get("ORCHESTRATOR_URL", "http://127.0.0.1:41246")

    async with httpx.AsyncClient(timeout=240.0) as http:
        card = await A2ACardResolver(
            httpx_client=http, base_url=orch_url
        ).get_agent_card()
        client = A2AClient(agent_card=card, httpx_client=http)

        payload = {
            "message": {
                "role": "user",
                "messageId": uuid4().hex,
                "parts": [
                    {
                        "kind": "data",
                        "data": {
                            "image_path": str(image_path),
                            "locale_hint": locale_hint,
                            "patient_lang_target": target_lang,
                        },
                    }
                ],
            }
        }
        req = SendMessageRequest(id=str(uuid4()), params=MessageSendParams(**payload))
        resp = await client.send_message(req)
        dumped = resp.model_dump(mode="json", exclude_none=True)

        if "error" in dumped and dumped["error"]:
            return {"_error": dumped["error"]}

        parts = (
            dumped.get("result", {}).get("parts")
            or dumped.get("result", {}).get("message", {}).get("parts")
            or []
        )
        for p in parts:
            # accept both shapes: {"type":"data"} or {"kind":"data"}
            if ("data" in p) and (p.get("type") == "data" or p.get("kind") == "data"):
                return p["data"]

        # Fallback for debugging if no DataPart found
        return {"_raw": dumped}


# CLI
async def launch_and_run(image: Path, locale: str, target: str) -> int:
    procs: List[asyncio.subprocess.Process] = []
    try:
        procs = await launch_all()
        print("\n[client] calling Orchestrator…")
        result = await send_to_orchestrator(image, locale, target)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        print(
            "[orchestrator] Workflow completed successfully.\nReferral packet generated:"
        )
        final_message = result.get("final_message")
        if final_message:
            print(final_message)
        return 0
    except KeyboardInterrupt:
        print("\n[client] interrupted.")
        return 130
    except Exception as e:
        print(f"[client] error: {e}")
        return 1
    finally:
        await cleanup(procs)


async def call_only(image: Path, locale: str, target: str) -> int:
    try:
        await wait_ready("http://127.0.0.1:41246")
        result = await send_to_orchestrator(image, locale, target)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except Exception as e:
        print(f"[client] error: {e}")
        return 1


# Entry
def main() -> None:
    parser = argparse.ArgumentParser(description="MedicalPassport launcher/client")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_launch = sub.add_parser(
        "launch", help="Start all agents, then call Orchestrator once and exit"
    )
    p_launch.add_argument(
        "image", type=Path, nargs="?", default=Path("data/samples/note_ar.jpg")
    )
    p_launch.add_argument(
        "--locale", default="ara", help="Tesseract OCR hint (e.g., eng, ara)"
    )
    p_launch.add_argument(
        "--target", default="en", help="Working language (default: en)"
    )

    p_call = sub.add_parser(
        "call", help="Call an already-running Orchestrator once and exit"
    )
    p_call.add_argument(
        "image", type=Path, nargs="?", default=Path("data/samples/note_ar.jpg")
    )
    p_call.add_argument(
        "--locale", default="ara", help="Tesseract OCR hint (e.g., eng, ara)"
    )
    p_call.add_argument("--target", default="en", help="Working language (default: en)")

    args = parser.parse_args()

    if args.cmd == "launch":
        code = asyncio.run(launch_and_run(args.image, args.locale, args.target))
    else:
        code = asyncio.run(call_only(args.image, args.locale, args.target))

    sys.exit(code)


if __name__ == "__main__":
    main()
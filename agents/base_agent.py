from __future__ import annotations
import json
from mcp import ClientSession
from a2a.types import AgentCard
import logging, json

logger = logging.getLogger("medicalpassport")
if not logger.handlers:
    h = logging.StreamHandler()
    fmt = logging.Formatter("[%(levelname)s] %(name)s: %(message)s")
    h.setFormatter(fmt)
    logger.addHandler(h)
    logger.setLevel(logging.INFO)


class BaseAgent:
    """Base class for all agents"""

    def __init__(self, session: ClientSession, card: AgentCard):
        self.session = session
        self.card = card
        self._tool_cache: set[str] | None = None

    def ensure_skill(self, skill_id: str) -> None:
        """Verify declared AgentSkill is present on the AgentCard."""
        if not any(s.id == skill_id for s in getattr(self.card, "skills", [])):
            raise RuntimeError(f"required skill '{skill_id}' missing from AgentCard")

    async def ensure_tool(self, tool_name: str) -> None:
        """Cache advertised tools + verify required tool availability."""
        if self._tool_cache is None:
            listing = await self.session.list_tools()
            self._tool_cache = {t.name for t in getattr(listing, "tools", [])}
        if tool_name not in self._tool_cache:
            raise RuntimeError(
                f"required tool '{tool_name}' is not advertised by the server"
            )

    async def call_tool_logged(self, tool_name: str, params: dict):
        await self.ensure_tool(tool_name)
        agent_name = getattr(self.card, "name", self.__class__.__name__)
        logger.info(
            "[MCP] %s -> %s payload=%s",
            agent_name,
            tool_name,
            json.dumps(params, ensure_ascii=False),
        )
        res = await self.session.call_tool(tool_name, params)
        try:
            data = self.extract_json(tool_name, res)
            keys = list(data.keys()) if isinstance(data, dict) else type(data).__name__
            logger.info("[MCP] %s <- %s result_keys=%s", agent_name, tool_name, keys)
        except Exception as e:
            logger.info("[MCP] %s <- %s nonjson err=%s", agent_name, tool_name, e)
        return res

    @staticmethod
    def extract_json(label: str, result):
        """Extract JSON content from an MCP tool result."""
        parts = getattr(result, "content", None) or []
        for p in parts:
            if getattr(p, "type", "") == "application/json" and hasattr(p, "data"):
                return p.data
            if hasattr(p, "text") and isinstance(p.text, str):
                try:
                    return json.loads(p.text)
                except Exception:
                    pass
        raise RuntimeError(f"{label} returned no JSON content")

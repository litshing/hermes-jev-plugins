"""jev-memory-gate — a Jev judgement in front of the `memory` tool.

Why
---
The agent writes its own long-term memory, and it is the only judge of what
deserves to survive. That goes wrong in both directions: trivia accumulates
until the store is full (memory is injected into EVERY turn, so a wasted entry
is a permanent tax), and the pruning to fix it is done under pressure.

This plugin puts a second opinion in front of the write. It asks TypeSafe's Jev
model whether a candidate entry is worth carrying across sessions — durable,
non-obvious, and steering-reducing. Clear filler is rejected without being
written; anything genuinely useful passes straight through.

Contract
--------
``tool_execution`` middleware (see ``docs/middleware/README.md``). Execution
middleware receives ``next_call`` and may short-circuit deliberately, which is
how a write is declined: return a tool result instead of calling ``next_call``.
It runs after ``tool_request`` and approval checks, so nothing upstream is
subverted — the call is simply answered differently.

Safety
------
* Fail-open at every step. Any error, missing key, timeout or malformed answer
  calls ``next_call`` and the write proceeds exactly as before.
* Conservative by default. Only a *confident* rejection blocks; the default
  threshold sits well below "probably filler" so a borderline entry is written.
* Never touches reads. Only ``add`` and ``replace`` are judged.
"""

from __future__ import annotations

import json
import logging
import os
import ssl
import threading
import time
import urllib.request
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

ENDPOINT = "https://api.typesafe.ai/v1/systemone"
ENV_FILE = os.path.expanduser("~/.hermes/.env")
DEFAULT_MODEL = "jev-latest"

PLUGIN_NAME = "jev-memory-gate"

DEFAULT_CONFIG: Dict[str, Any] = {
    "enabled": True,
    "model": DEFAULT_MODEL,
    # Below this probability of being worth keeping, the write is declined.
    # Deliberately low: blocking a real preference costs more than storing a
    # mediocre entry, because the user has to notice and re-state it.
    "block_below": 0.25,
    # Never judge tiny entries (usually edits) or enormous ones.
    "min_chars": 40,
    "max_chars": 4000,
    "timeout_s": 45.0,
    "min_seconds_between_calls": 3.0,
    "error_cooldown_s": 120.0,
    "log_decisions": True,
}

JUDGED_ACTIONS = frozenset({"add", "replace"})


def _load_config() -> Dict[str, Any]:
    cfg = dict(DEFAULT_CONFIG)
    path = os.path.join(
        os.environ.get("HERMES_HOME") or os.path.expanduser("~/.hermes"), "config.yaml"
    )
    try:
        import yaml  # type: ignore
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        block = ((raw.get("plugins") or {}).get("jev_memory_gate") or {})
        if isinstance(block, dict):
            for k, v in block.items():
                if k in cfg and v is not None:
                    cfg[k] = v
    except Exception:
        pass
    return cfg


def _load_key() -> Optional[str]:
    key = os.environ.get("TYPESAFE_API_KEY")
    if key:
        return key.strip()
    try:
        with open(ENV_FILE) as f:
            for line in f:
                if line.startswith("TYPESAFE_API_KEY="):
                    return line.strip().split("=", 1)[1].strip().strip('"').strip("'")
    except OSError:
        return None
    return None


def judge(entry: str, cfg: Dict[str, Any]) -> Optional[float]:
    """Probability that ``entry`` is worth carrying across sessions.

    Returns ``None`` on any failure — the caller treats that as "let it through".
    """
    state = {
        "candidate_memory_entry": entry[: int(cfg["max_chars"])],
        "entry_chars": len(entry),
    }
    questions = {
        "worth_remembering": {
            "type": "noul",
            "instructions": (
                "An AI agent is deciding whether to store this in its permanent, "
                "cross-session memory. That memory is injected into every future "
                "conversation, so an unworthy entry is a permanent tax on every "
                "turn and will eventually force something better out. Is this "
                "entry worth carrying across sessions?"
            ),
            "criteria": {
                "true": "A durable fact, stated preference, correction, environment detail, or convention that will still matter weeks from now and reduces repeated user steering — OR a change to an existing entry that keeps it accurate",
                "false": "Task progress, completed-work logs, session outcomes, restated context, ephemeral state, raw data dumps, anything trivially rediscoverable, or a near-duplicate of what is presumably already stored",
            },
        },
    }
    body = json.dumps(
        {"model": cfg.get("model") or DEFAULT_MODEL, "state": state, "questions": questions},
        ensure_ascii=False,
    ).encode()
    req = urllib.request.Request(
        ENDPOINT,
        data=body,
        headers={
            "Authorization": f"Bearer {_load_key()}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(
        req, timeout=float(cfg["timeout_s"]), context=ssl.create_default_context()
    ) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    value = ((payload.get("answers") or {}).get("worth_remembering") or {}).get("noul")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


class MemoryGate:
    """Holds the cooldown / error state the middleware callbacks share."""

    def __init__(self) -> None:
        self.cfg = _load_config()
        self._lock = threading.Lock()
        self._last_call = 0.0
        self._error_until = 0.0
        self.stats: Dict[str, int] = {
            "judged": 0, "passed": 0, "blocked": 0,
            "skipped_no_key": 0, "skipped_cooldown": 0, "errors": 0,
        }

    # -- helpers ---------------------------------------------------------

    @staticmethod
    def _candidates(args: Dict[str, Any]) -> List[tuple]:
        """Every (label, text) pair this call would write."""
        action = str(args.get("action") or "").lower()
        out: List[tuple] = []
        if action in JUDGED_ACTIONS:
            content = args.get("content")
            if isinstance(content, str) and content.strip():
                out.append((action, content))
        for op in args.get("operations") or []:
            if not isinstance(op, dict):
                continue
            if str(op.get("action") or "").lower() not in JUDGED_ACTIONS:
                continue
            content = op.get("content") or op.get("new_text")
            if isinstance(content, str) and content.strip():
                out.append(("batch:" + str(op.get("action")), content))
        return out

    def _should_judge(self, text: str) -> bool:
        n = len(text.strip())
        return int(self.cfg["min_chars"]) <= n <= int(self.cfg["max_chars"])

    def _budget_ok(self) -> bool:
        now = time.time()
        with self._lock:
            if now < self._error_until:
                return False
            if now - self._last_call < float(self.cfg["min_seconds_between_calls"]):
                self.stats["skipped_cooldown"] += 1
                return False
            self._last_call = now
        if not _load_key():
            self.stats["skipped_no_key"] += 1
            return False
        return True

    def _note_error(self) -> None:
        self.stats["errors"] += 1
        with self._lock:
            self._error_until = time.time() + float(self.cfg["error_cooldown_s"])

    # -- the decision ----------------------------------------------------

    def evaluate(self, args: Dict[str, Any]) -> Optional[str]:
        """Return a rejection message, or ``None`` to let the write proceed."""
        if not self.cfg.get("enabled", True):
            return None
        candidates = [c for c in self._candidates(args) if self._should_judge(c[1])]
        if not candidates:
            return None
        if not self._budget_ok():
            return None

        verdicts = []
        for label, text in candidates:
            try:
                p = judge(text, self.cfg)
            except Exception as exc:
                self._note_error()
                logger.warning("jev-memory-gate: judgement failed, allowing write: %s", exc)
                return None
            if p is None:
                return None
            self.stats["judged"] += 1
            verdicts.append((label, p, text))

        blocked = [
            v for v in verdicts if v[1] < float(self.cfg["block_below"])
        ]
        if not blocked:
            self.stats["passed"] += 1
            return None

        self.stats["blocked"] += 1
        lines = [
            f"NOT SAVED — judged not worth permanent memory (p={p:.2f} < "
            f"{float(self.cfg['block_below']):.2f})."
            for _, p, _ in blocked
        ]
        detail = "; ".join(f"{label}: p={p:.2f}" for label, p, _ in blocked)
        if self.cfg.get("log_decisions", True):
            logger.info("jev-memory-gate: declined write (%s)", detail)
        return json.dumps(
            {
                "ok": False,
                "saved": False,
                "reason": " ".join(lines),
                "detail": detail,
                "guidance": (
                    "This entry reads like ephemeral or rediscoverable content "
                    "(task progress, session output, restated context). Permanent "
                    "memory is injected into every future conversation, so it is "
                    "reserved for durable preferences, corrections, environment "
                    "facts and conventions. If you still believe this must be "
                    "remembered, say so explicitly to the user rather than "
                    "retrying silently."
                ),
            },
            ensure_ascii=False,
        )


_GATE: Optional[MemoryGate] = None
_GATE_LOCK = threading.Lock()


def _gate() -> MemoryGate:
    global _GATE
    with _GATE_LOCK:
        if _GATE is None:
            _GATE = MemoryGate()
        return _GATE


def on_tool_execution(**kwargs: Any):
    """``tool_execution`` middleware: judge the write, else delegate.

    This callback sits in the chain for EVERY tool call, so the delegation path
    must be exact. ``hermes_cli/middleware.py::_run_execution_chain`` does
    ``return callback(**call_kwargs)`` — a callback that returns ``None`` without
    calling ``next_call`` hands ``None`` back as the tool result. That would
    silently break every tool in the session, so every non-judged path returns
    ``next_call(args)`` and nothing else.
    """
    next_call = kwargs.get("next_call")
    args = kwargs.get("args")
    if not callable(next_call):
        return None
    # Anything that is not a judged memory write is a pure pass-through.
    if kwargs.get("tool_name") != "memory" or not isinstance(args, dict):
        return next_call(args)

    try:
        rejection = _gate().evaluate(dict(args))
    except Exception as exc:  # fail-open, always
        logger.warning("jev-memory-gate: gate error, allowing write: %s", exc)
        rejection = None

    if rejection is not None:
        return rejection  # deliberate short-circuit: next_call is NOT called
    return next_call(args)


def register(ctx) -> None:
    ctx.register_middleware("tool_execution", on_tool_execution)

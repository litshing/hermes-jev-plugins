"""Jev decision client for context compaction.

Talks to TypeSafe's System One endpoint (model ``jev-latest``) with the same
protocol already used by ``~/antiques/typesafe_gate.py`` — the only difference
is what we put in ``state`` (a conversation instead of an auction lot) and what
the questions ask.

Design rules (mirroring typesafe_gate.py):
  * Never raise into the caller. Any failure degrades to an empty decision set,
    so a broken key or a network blip leaves the transcript untouched.
  * Text only, and only as much of it as is needed to judge a decision.
  * Decide per tool call: keep it and its result verbatim, keep the call but
    shrink the result, or drop the pair entirely.
"""

from __future__ import annotations

import hashlib
import json
import os
import ssl
import threading
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Sequence, Tuple

ENDPOINT = "https://api.typesafe.ai/v1/systemone"
ENV_FILE = os.path.expanduser("~/.hermes/.env")
DEFAULT_MODEL = "jev-latest"

KEEP = "keep"
TRUNCATE = "truncate"
DROP = "drop"

# Tool results whose loss the model can observe directly, and which must never
# be dropped wholesale: dropping a skill body is the "ghost skill" failure
# (#32106) — the model believes instructions are still in context.
NEVER_DROP_TOOLS = frozenset({"skill_view"})


# ---------------------------------------------------------------- key

def load_key() -> Optional[str]:
    """TYPESAFE_API_KEY from the environment, else ~/.hermes/.env."""
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


# ------------------------------------------------------- token estimate

_WORD_CHARS = 6


def estimate_tokens(text: str) -> int:
    """Deliberately rough estimate, no tokenizer.

    Letters cost ~1 token per 6 characters, digits ~0.5 each, everything else
    ~1 each. Calibrated to land a little above real counts, so a budget check
    errs on the side of sending less.
    """
    if not text:
        return 0
    letters = digits = other = 0
    for ch in text:
        if ch.isalpha():
            letters += 1
        elif ch.isdigit():
            digits += 1
        else:
            other += 1
    return int(letters / _WORD_CHARS + digits * 0.5 + other) + 1


def estimate_messages_tokens(messages: Sequence[Dict[str, Any]]) -> int:
    total = 0
    for m in messages:
        content = m.get("content")
        if isinstance(content, str):
            total += estimate_tokens(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    total += estimate_tokens(part["text"])
        for tc in m.get("tool_calls") or []:
            fn = (tc or {}).get("function") or {}
            total += estimate_tokens(str(fn.get("arguments", "")))
        total += 4
    return total


# ----------------------------------------------------- message walking

def _index_calls(messages: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """tool_call_id -> {name, args, assistant_idx, result_idx, result_text}."""
    calls: Dict[str, Dict[str, Any]] = {}
    for i, msg in enumerate(messages):
        if msg.get("role") != "assistant":
            continue
        for tc in msg.get("tool_calls") or []:
            if not isinstance(tc, dict):
                continue
            cid = tc.get("id") or ""
            fn = tc.get("function") or {}
            calls[cid] = {
                "id": cid,
                "name": fn.get("name", "unknown"),
                "args": fn.get("arguments", ""),
                "assistant_idx": i,
                "result_idx": None,
                "result_text": "",
            }
    for i, msg in enumerate(messages):
        if msg.get("role") != "tool":
            continue
        cid = msg.get("tool_call_id") or ""
        entry = calls.get(cid)
        if entry is None:
            continue
        content = msg.get("content")
        if isinstance(content, str):
            entry["result_idx"] = i
            entry["result_text"] = content
    return {cid: e for cid, e in calls.items() if e["result_idx"] is not None}


def iter_candidates(
    messages: Sequence[Dict[str, Any]],
    protect_first_n: int,
    protect_last_n: int,
    min_result_chars: int,
) -> List[Dict[str, Any]]:
    """Tool calls eligible for a Jev decision.

    Pinned by construction: the first ``protect_first_n`` non-system messages
    and the newest ``protect_last_n`` messages are never offered as candidates,
    so a decision can never touch the active exchange.
    """
    n = len(messages)
    # Resolve protected head/tail to index ranges.
    head_end = 0
    seen = 0
    for i, m in enumerate(messages):
        if m.get("role") == "system":
            head_end = i + 1
            continue
        seen += 1
        head_end = i + 1
        if seen >= protect_first_n:
            break
    tail_start = max(head_end, n - protect_last_n)

    out: List[Dict[str, Any]] = []
    for entry in _index_calls(messages).values():
        a_i = entry["assistant_idx"]
        r_i = entry["result_idx"]
        if a_i < head_end or r_i < head_end or a_i >= tail_start or r_i >= tail_start:
            continue
        if len(entry["result_text"]) <= min_result_chars:
            continue
        out.append(entry)
    out.sort(key=lambda e: e["assistant_idx"])
    return out


# ------------------------------------------------------------- state

def _tool_note(entry: Dict[str, Any]) -> str:
    return f"ok, {len(entry['result_text'])} chars (omitted)"


def build_state(
    messages: Sequence[Dict[str, Any]],
    candidates: Sequence[Dict[str, Any]],
    goal: str,
    max_state_tokens: int,
    max_input_chars: int = 1000,
) -> Dict[str, Any]:
    """The conversation Jev judges, oldest first.

    Every tool result is replaced by a short note (its contents are what we are
    asking *about*, not what we show). Tool inputs and message texts are
    included. Older messages are abridged head+tail first, then dropped from the
    front, until the estimated size fits ``max_state_tokens``.
    """
    cand_ids = {c["id"] for c in candidates}
    entries: List[Dict[str, Any]] = []
    for m in messages:
        role = m.get("role")
        if role == "system":
            continue
        text = m.get("content") if isinstance(m.get("content"), str) else ""
        item: Dict[str, Any] = {"role": role}
        if text:
            item["text"] = text
        if role == "assistant":
            tcs = []
            for tc in m.get("tool_calls") or []:
                if not isinstance(tc, dict):
                    continue
                fn = tc.get("function") or {}
                args = str(fn.get("arguments", ""))
                if len(args) > max_input_chars:
                    args = args[:max_input_chars] + f"…[{len(args)} chars total]"
                tcs.append({"name": fn.get("name", "unknown"), "input": args})
            if tcs:
                item["tool_calls"] = tcs
        if role == "tool":
            cid = m.get("tool_call_id") or ""
            item["tool_call_id"] = cid
            item["result"] = _tool_note(
                {"result_text": m.get("content") if isinstance(m.get("content"), str) else ""}
            )
        entries.append(item)

    def total(items: Sequence[Dict[str, Any]]) -> int:
        return estimate_tokens(json.dumps(items, ensure_ascii=False))

    # Stage 1: abridge the text of old messages not tied to a candidate.
    if total(entries) > max_state_tokens:
        keep_from = len(entries) - 40
        for i, item in enumerate(entries):
            if i >= keep_from:
                continue
            if item.get("tool_call_id") in cand_ids:
                continue
            t = item.get("text")
            if isinstance(t, str) and len(t) > 400:
                item["text"] = t[:240] + "…" + t[-120:]
    # Stage 2: drop the oldest entries until it fits.
    while entries and total(entries) > max_state_tokens:
        entries.pop(0)

    return {"goal": goal, "messages": entries}


def build_questions(candidates: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    questions: Dict[str, Any] = {}
    for c in candidates:
        arg_repr = c["args"]
        if len(arg_repr) > 300:
            arg_repr = arg_repr[:300] + "…"
        questions[f"keep_call_{c['id']}"] = {
            "type": "noul",
            "instructions": (
                f"Earlier in this session the assistant called {c['name']} with "
                f"input {arg_repr}. Is retaining that this call happened still "
                f"relevant to the work in progress?"
            ),
            "criteria": {
                "true": "The call and its input still matter — they pin a decision, a constraint, or a file/entity under work",
                "false": "The call is superseded, redundant, or no longer relevant to what is being done now",
            },
        }
        questions[f"keep_result_{c['id']}"] = {
            "type": "noul",
            "instructions": (
                f"The output of that {c['name']} call was {len(c['result_text'])} "
                f"characters. Are its contents still needed verbatim later, such "
                f"that re-running the tool would not serve just as well?"
            ),
            "criteria": {
                "true": "Exact contents are still needed later — error text, line numbers, values, or a file being edited",
                "false": "The contents are stale, already superseded by a later result, or cheaply reproducible by re-running the tool",
            },
        }
    return questions


# ------------------------------------------------------------ transport

def ask(
    state: Dict[str, Any],
    questions: Dict[str, Any],
    *,
    key: str,
    model: str = DEFAULT_MODEL,
    timeout: float = 90.0,
) -> Dict[str, Any]:
    """One Jev request. Raises on transport/HTTP errors — callers fail open."""
    body = json.dumps(
        {"model": model, "state": state, "questions": questions}, ensure_ascii=False
    ).encode()
    req = urllib.request.Request(
        ENDPOINT,
        data=body,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        method="POST",
    )
    ctx = ssl.create_default_context()
    with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _noul(answers: Dict[str, Any], name: str) -> Optional[float]:
    entry = answers.get(name)
    if not isinstance(entry, dict):
        return None
    value = entry.get("noul")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def decide(
    answers: Dict[str, Any],
    candidates: Sequence[Dict[str, Any]],
    keep_threshold: float,
) -> Tuple[Dict[str, str], int]:
    """Map each candidate to keep / truncate / drop.

    Returns ``(decisions, n_missing)``. Candidates with no usable answer are
    omitted entirely (treated as keep by the caller).
    """
    decisions: Dict[str, str] = {}
    missing = 0
    for c in candidates:
        cid = c["id"]
        keep_result = _noul(answers, f"keep_result_{cid}")
        keep_call = _noul(answers, f"keep_call_{cid}")
        if keep_result is None or keep_call is None:
            missing += 1
            continue
        if keep_result >= keep_threshold:
            decisions[cid] = KEEP
        elif keep_call >= keep_threshold:
            decisions[cid] = TRUNCATE
        else:
            decisions[cid] = DROP
    return decisions, missing


# -------------------------------------------------------------- cache

class DecisionCache:
    """In-process memo so an unchanged pair is never judged twice.

    Keyed on the call id plus the size of the result — if a tool result has
    not changed, a second opinion costs money and buys nothing. Decisions do
    not outlive the process, because they depend on a conversation that has.
    """

    def __init__(self, max_entries: int = 2000):
        self._lock = threading.Lock()
        self._data: Dict[str, str] = {}
        self._max = max_entries

    def _key(self, c: Dict[str, Any]) -> str:
        digest = hashlib.sha1(
            f"{c['name']}|{c['args'][:400]}|{len(c['result_text'])}".encode(
                "utf-8", errors="replace"
            )
        ).hexdigest()[:16]
        return digest

    def get_many(self, candidates: Sequence[Dict[str, Any]]) -> Dict[str, str]:
        with self._lock:
            return {
                c["id"]: self._data[self._key(c)]
                for c in candidates
                if self._key(c) in self._data
            }

    def put_many(self, candidates: Sequence[Dict[str, Any]], decisions: Dict[str, str]) -> None:
        with self._lock:
            for c in candidates:
                d = decisions.get(c["id"])
                if d:
                    self._data[self._key(c)] = d
            while len(self._data) > self._max:
                self._data.pop(next(iter(self._data)))

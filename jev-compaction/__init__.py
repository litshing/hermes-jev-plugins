"""jev-compaction — a context engine that prunes by *need* instead of by *size*.

Why this exists
---------------
Hermes' built-in ``ContextCompressor`` clears old tool output deterministically:
anything over ``proactive_prune_min_result_chars`` that has aged out of the
protected tail gets replaced by a one-line summary. That rule is blind to
whether the output still matters. It will happily keep a 40-char result that a
later decision depended on, and clear a 30 KB result the model is about to need
again.

This engine keeps the built-in's plumbing — the deterministic dedup pass, the
prompt-cache hysteresis, the ``archive_and_compact`` session persistence, the
re-arm runway — and inserts one thing in front of it: a per-tool-call judgement
from TypeSafe's Jev model (``jev-latest``) that says, for each old tool call,
whether the call still matters, and whether its *contents* are still needed
verbatim. Calls that no longer matter lose the result *body* but keep their row
(a one-line note takes its place, and the call itself survives); calls that
matter but whose contents don't keep the call and lose the body.

ROW COUNT IS A HARD CONTRACT
----------------------------
``ContextCompressor.compress`` captures ``n_messages = len(messages)`` *before*
calling the ``_prune_old_tool_results`` seam, and only recomputes it when it
strips platform-echo rows itself. A prune that returns a shorter list therefore
leaves the assembled tail loop (``for i in range(..., n_messages)``) indexing
past the end of the list it now walks → ``IndexError: list index out of range``
inside ``context_compressor.compress``. Observed live on 2026-09-18: 80 dropped
messages, 414-message session, every auto-compression crashed. This engine
therefore never adds or removes a row: DROP empties a payload, it does not
delete a message.

Nothing is ever summarized. Text, user turns and assistant reasoning pass
through untouched.

Contract notes
--------------
* Subclasses ``ContextCompressor``, so it inherits every abstract method the
  ``ContextEngine`` ABC requires, plus the tested persistence path. Worst case
  (no key, network error, malformed answer, timeout) it behaves *exactly* like
  the built-in.
* ``_prune_old_tool_results`` is the single seam. It is called on both the
  cheap proactive-prune path and the full-compression path, so one override
  covers both.
* Fail-open everywhere: any exception is logged and the original list is
  handed to the built-in unchanged.
* Thread-safe: ``compress()`` may be dispatched on a pooled daemon thread, so
  the decision cache and the call cooldown are lock-guarded.
"""

from __future__ import annotations

import copy as _copy
import json
import logging
import os
import threading
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

from agent.context_compressor import ContextCompressor, _PRUNE_MIN_CHARS

from . import jev

PLUGIN_NAME = "jev-compaction"

DEFAULT_CONFIG: Dict[str, Any] = {
    "enabled": True,
    "model": jev.DEFAULT_MODEL,
    "keep_threshold": 0.5,
    # Never offered as candidates — mirrors the ABC's protect_first_n/protect_last_n.
    "protect_first_n": 3,
    "protect_last_n": 20,
    # A candidate result must be at least this big to be worth judging.
    "min_result_chars": _PRUNE_MIN_CHARS,
    # Skip the request entirely for a thin batch; a Jev call has a floor cost.
    "min_candidates": 4,
    # Don't re-ask more often than this (seconds) within one process.
    "min_seconds_between_calls": 45.0,
    "max_calls_per_pass": 40,
    "max_state_tokens": 25000,
    "truncate_head_chars": 300,
    "timeout_s": 90.0,
}


def _load_config() -> Dict[str, Any]:
    """``context.jev`` from config.yaml, over the defaults.

    Best-effort and non-fatal: a missing/unreadable config just means defaults.
    """
    cfg = dict(DEFAULT_CONFIG)
    path = os.path.join(os.environ.get("HERMES_HOME") or os.path.expanduser("~/.hermes"),
                        "config.yaml")
    try:
        import yaml  # type: ignore
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        block = ((raw.get("context") or {}).get("jev") or {})
        if isinstance(block, dict):
            for k, v in block.items():
                if k in cfg and v is not None:
                    cfg[k] = v
    except Exception:
        pass
    return cfg


def _cfg_num(cfg: Dict[str, Any], key: str, default, cast, minimum=None):
    """Read a numeric config value, keeping an explicit 0.

    ``cfg.get(key) or default`` silently rewrites a deliberate 0 — and 0 is a
    meaningful value for several of these knobs (protect nothing, keep only the
    note, never skip a thin batch). Treat only None/missing/unparseable as
    "unset", and clamp instead of swallowing.
    """
    value = cfg.get(key, None)
    if value is None:
        return default
    try:
        value = cast(value)
    except (TypeError, ValueError):
        return default
    if minimum is not None and value < minimum:
        return default if default >= minimum else minimum
    return value


def _cfg_int(cfg: Dict[str, Any], key: str, default: int, minimum: int = 0) -> int:
    return int(_cfg_num(cfg, key, default, int, minimum))


def _cfg_float(cfg: Dict[str, Any], key: str, default: float, minimum: float = 0.0) -> float:
    return float(_cfg_num(cfg, key, default, float, minimum))


# ------------------------------------------------------------------ rewrite

_DROP_NOTE_MARK = "dropped by jev-compaction"


def _drop_note(content: Any) -> str:
    """The one-line stand-in that replaces a dropped result body."""
    size = len(content) if isinstance(content, str) else 0
    return (
        f"[{_DROP_NOTE_MARK}: {size:,} chars judged no longer relevant and set "
        f"aside; re-run the tool if you need it again]"
    )


def _drop_plan(messages: Sequence[Dict[str, Any]],
               decisions: Dict[str, str]) -> Tuple[List[Dict[str, Any]], int]:
    """Apply keep / truncate / drop decisions to a message list.

    A DROP empties the *payload*, never the row: the tool result keeps its
    position and its ``tool_call_id``, and its body becomes a one-line note.
    The call itself is always kept, so a result can never be orphaned.

    ``len(out) == len(messages)`` is a hard contract, not cosmetics —
    ``ContextCompressor.compress`` reads ``n_messages`` *before* the prune seam
    and does not recompute it, so returning a shorter list makes its tail
    assembly loop index past the end (``IndexError`` in the compressor). See
    "ROW COUNT IS A HARD CONTRACT" in the module docstring. Keeping the row
    count also keeps ``_truncate_results``' index-based writes pointing at the
    right rows.

    Guarantees preserved:
      * the row count never changes;
      * a tool result is never left without its call, and a call never without
        its result (the call is always kept);
      * everything not explicitly dropped is returned as the same object.
    """
    if not decisions:
        return list(messages), 0

    dropped_results = {cid for cid, d in decisions.items() if d == jev.DROP}
    changed = 0
    out: List[Dict[str, Any]] = []

    for msg in messages:
        role = msg.get("role")

        if role == "tool" and (msg.get("tool_call_id") or "") in dropped_results:
            content = msg.get("content")
            already_set_aside = (
                isinstance(content, str) and _DROP_NOTE_MARK in content
            )
            if not already_set_aside:
                changed += 1
                msg = {**msg, "content": _drop_note(content)}
            out.append(msg)
            continue

        out.append(msg)

    return out, changed


def _truncate_results(messages: List[Dict[str, Any]],
                      candidates: Sequence[Dict[str, Any]],
                      decisions: Dict[str, str],
                      head_chars: int) -> Tuple[List[Dict[str, Any]], int]:
    """Replace the body of a TRUNCATE'd result with its head plus a note.

    The call survives in full, so the model still knows the action happened and
    with what arguments; only the bulky payload goes, and the note says the tool
    can be re-run.
    """
    by_result_idx = {
        c["result_idx"]: c for c in candidates if decisions.get(c["id"]) == jev.TRUNCATE
    }
    if not by_result_idx:
        return messages, 0
    changed = 0
    for idx, c in by_result_idx.items():
        if idx >= len(messages):
            continue
        msg = messages[idx]
        content = msg.get("content")
        if not isinstance(content, str) or not content:
            continue
        head = content[:head_chars]
        note = (f"\n…[{len(content)} chars total — oldest part of this {c['name']} "
                f"output set aside to save context; re-run {c['name']} if you need "
                f"the full output]")
        messages[idx] = {**msg, "content": head + note}
        changed += 1
    return messages, changed


# ------------------------------------------------------------------ engine

class JevContextCompressor(ContextCompressor):
    """ContextCompressor + a Jev judgement in front of the prune pass."""

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._jev_cfg = _load_config()
        self._jev_cache = jev.DecisionCache()
        self._jev_lock = threading.Lock()
        self._jev_last_call = 0.0
        self._jev_stats: Dict[str, int] = {
            "passes": 0, "requests": 0, "judged": 0,
            "dropped": 0, "truncated": 0, "kept": 0, "errors": 0, "skipped": 0,
        }

    @property
    def name(self) -> str:
        return PLUGIN_NAME

    # -- host contract: the engine gets deep-copied -----------------------

    def __deepcopy__(self, memo: Dict[int, Any]) -> "JevContextCompressor":
        """Copy budget state, but never the lock or the memo cache.

        The host deep-copies the shared plugin singleton per agent
        (``agent_init.py``, #42449) so one agent's ``update_model()`` cannot
        mutate another's compressor. A ``threading.Lock`` inside the copy would
        raise, and the host would then silently fall back to the built-in
        compressor — this engine would appear to be installed and quietly do
        nothing. So locks and caches are always rebuilt fresh, and anything
        genuinely uncopyable is shared rather than dropped.
        """
        cls = self.__class__
        new = cls.__new__(cls)
        memo[id(self)] = new
        for key, value in self.__dict__.items():
            if key == "_jev_lock":
                setattr(new, key, threading.Lock())
            elif key == "_jev_cache":
                setattr(new, key, jev.DecisionCache())
            else:
                try:
                    setattr(new, key, _copy.deepcopy(value, memo))
                except Exception:
                    setattr(new, key, value)
        return new

    # -- the seam ---------------------------------------------------------

    def _prune_old_tool_results(
        self,
        messages: List[Dict[str, Any]],
        protect_tail_count: int,
        protect_tail_tokens: int | None = None,
        min_prune_chars: int = _PRUNE_MIN_CHARS,
    ) -> Tuple[List[Dict[str, Any]], int]:
        """Jev pre-pass, then the built-in's deterministic passes.

        Ordering matters: the drop/truncate decisions run first so the built-in
        then works on an already-reduced list and its size-based rules pick off
        whatever is left. Its dedup, hysteresis and persistence behaviour are
        untouched.
        """
        pre, jev_changes = self._jev_prepass(messages, protect_tail_count)
        out, pruned = super()._prune_old_tool_results(
            pre, protect_tail_count, protect_tail_tokens, min_prune_chars
        )
        return out, pruned + jev_changes

    # -- the judgement ----------------------------------------------------

    def _jev_prepass(
        self, messages: List[Dict[str, Any]], protect_tail_count: int,
    ) -> Tuple[List[Dict[str, Any]], int]:
        """Returns ``(messages_or_original, n_changes)``; never raises."""
        cfg = self._jev_cfg
        if not cfg.get("enabled", True) or not messages:
            return messages, 0

        try:
            return self._jev_prepass_inner(messages, protect_tail_count, cfg)
        except Exception as exc:  # fail open — never break a compaction pass
            self._jev_stats["errors"] += 1
            logger.warning("jev-compaction: pre-pass failed, deferring to built-in: %s", exc)
            return messages, 0

    def _jev_prepass_inner(
        self, messages: List[Dict[str, Any]], protect_tail_count: int,
        cfg: Dict[str, Any],
    ) -> Tuple[List[Dict[str, Any]], int]:
        protect_last = max(_cfg_int(cfg, "protect_last_n", 20), protect_tail_count)
        candidates = jev.iter_candidates(
            messages,
            protect_first_n=_cfg_int(cfg, "protect_first_n", 3),
            protect_last_n=protect_last,
            min_result_chars=_cfg_int(cfg, "min_result_chars", _PRUNE_MIN_CHARS),
        )
        # Never drop a skill body: the model would keep believing its
        # instructions are still in context (the "ghost skill" failure, #32106).
        candidates = [c for c in candidates if c["name"] not in jev.NEVER_DROP_TOOLS]
        min_cand = _cfg_int(cfg, "min_candidates", 4)
        if len(candidates) < min_cand:
            # Log this. A silent skip here is indistinguishable from a healthy
            # engine that simply had nothing to do, which is exactly how a
            # "why is Jev never firing?" hunt wastes an afternoon. The usual
            # cause is a model that batches every tool call into ONE early
            # assistant message: that message sits inside the protected head,
            # so all of its calls are excluded by construction.
            self._jev_stats["skipped"] += 1
            self._jev_last_skip = (
                f"only {len(candidates)} candidate(s), need min_candidates="
                f"{min_cand} (protect_first_n={_cfg_int(cfg, 'protect_first_n', 3)}, "
                f"protect_last_n={protect_last}, messages={len(messages)})"
            )
            logger.info("jev-compaction: pre-pass skipped — %s", self._jev_last_skip)
            return messages, 0

        decisions = self._jev_cache.get_many(candidates)
        fresh = [c for c in candidates if c["id"] not in decisions]
        if fresh:
            now = time.time()
            with self._jev_lock:
                wait = _cfg_float(cfg, "min_seconds_between_calls", 45.0) - (now - self._jev_last_call)
                if wait > 0:
                    self._jev_stats["skipped"] += 1
                    fresh = []  # cooldown: use whatever the cache already knows
                else:
                    self._jev_last_call = now
            if fresh:
                fresh = fresh[:_cfg_int(cfg, "max_calls_per_pass", 40)]
                answers, n_missing = self._ask_jev(messages, fresh, cfg)
                if answers is None:
                    self._jev_stats["skipped"] += 1
                else:
                    got, missing = jev.decide(
                        answers, fresh, _cfg_float(cfg, "keep_threshold", 0.5)
                    )
                    self._jev_stats["requests"] += 1
                    self._jev_stats["judged"] += len(got)
                    self._jev_cache.put_many(fresh, got)
                    decisions.update(got)

        if not decisions:
            self._jev_stats["skipped"] += 1
            return messages, 0

        self._jev_stats["passes"] += 1
        for d in decisions.values():
            if d == jev.DROP:
                self._jev_stats["dropped"] += 1
            elif d == jev.TRUNCATE:
                self._jev_stats["truncated"] += 1
            else:
                self._jev_stats["kept"] += 1

        # Splitting and byte-identical duplication must not shrink this set:
        # `_index_calls` keys by tool_call_id so each decision lands once.
        working = list(messages)
        working, n_drop = _drop_plan(working, decisions)
        working, n_trunc = _truncate_results(
            working, candidates, decisions, _cfg_int(cfg, "truncate_head_chars", 300)
        )
        total = n_drop + n_trunc
        if total and not self.quiet_mode:
            logger.info(
                "jev-compaction: %d decision(s) — %d dropped, %d truncated "
                "(threshold %.2f, cache %d)",
                total, n_drop, n_trunc,
                _cfg_float(cfg, "keep_threshold", 0.5),
                len(decisions),
            )
        return (working if total else messages), total

    def _ask_jev(
        self, messages: Sequence[Dict[str, Any]], fresh: Sequence[Dict[str, Any]],
        cfg: Dict[str, Any],
    ) -> Tuple[Optional[Dict[str, Any]], int]:
        """One request. Returns ``({answers}, n_missing)`` or ``(None, 0)``."""
        # Defect guard: an empty candidate list used to sail through to the API
        # and come back as a bare HTTP 422, which the caller saw as "no answers"
        # rather than "we asked nothing". Bail out before spending a request.
        if not fresh:
            self._jev_stats["skipped"] += 1
            return None, 0
        key = jev.load_key()
        if not key:
            self._jev_stats["skipped"] += 1
            return None, 0
        goal = self._goal(messages)
        state = jev.build_state(
            messages, fresh, goal,
            max_state_tokens=int(cfg.get("max_state_tokens") or 25000),
        )
        questions = jev.build_questions(fresh)
        resp = jev.ask(
            state, questions,
            key=key,
            model=str(cfg.get("model") or jev.DEFAULT_MODEL),
            timeout=float(cfg.get("timeout_s") or 90),
        )
        answers = resp.get("answers")
        if not isinstance(answers, dict):
            return None, 0
        return answers, 0

    @staticmethod
    def _goal(messages: Sequence[Dict[str, Any]]) -> str:
        """The latest user-authored turns, as the ongoing task description."""
        turns = [m.get("content") for m in messages
                 if m.get("role") == "user" and isinstance(m.get("content"), str)]
        return "\n".join(turns[-3:])[:2000]

    # -- observability ----------------------------------------------------

    def get_status(self) -> Dict[str, Any]:
        status = super().get_status()
        status["jev"] = dict(self._jev_stats)
        # Why the last pass did nothing, when it did nothing. Without this, a
        # skip is indistinguishable from a healthy engine that simply had no
        # work -- which is what made "Jev never fires" so expensive to chase.
        status["jev_last_skip"] = getattr(self, "_jev_last_skip", None)
        # The resolved knobs, so a reader can see WHICH window is in force
        # rather than guessing at defaults in two different config trees.
        status["jev_config"] = {
            k: self._jev_cfg.get(k) for k in (
                "enabled", "model", "keep_threshold", "protect_first_n",
                "protect_last_n", "min_candidates", "min_result_chars",
                "min_seconds_between_calls",
            )
        }
        return status


# ------------------------------------------------------------------ register

def register(ctx) -> None:
    """Register the engine. See developer-guide/context-engine-plugin.md."""
    engine = JevContextCompressor(model="")
    ctx.register_context_engine(engine)

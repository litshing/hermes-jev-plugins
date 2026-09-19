"""Contract tests for the jev-compaction engine.

Run with the Hermes venv, but NOT from the hermes-agent repo root:

    cd /tmp
    ~/.hermes/hermes-agent/venv/bin/python3 -m pytest \
        ~/.hermes/plugins/jev-compaction/tests/test_plugin.py -q

Running it from `~/.hermes/hermes-agent` (or from the plugin directory) makes
pytest's collection import the plugin's own `__init__.py` as a test module,
which fails on its relative `from . import jev` with
"attempted relative import with no known parent package" — every test then
reports as an ERROR before it runs. Verified: 28 passed / 1 skipped from /tmp,
28 errors from the repo root.

The live-network test is marked ``live`` and skipped unless
``JEV_LIVE=1`` is set, because it spends real (tiny) money.
"""

from __future__ import annotations

import copy
import importlib.util
import os
import sys

import pytest

REPO = os.path.expanduser("~/.hermes/hermes-agent")
PLUGIN_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)


def _load_plugin():
    """Import the plugin the way the loader does (hyphenated dir, no package name)."""
    name = "jev_compaction_under_test"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(
        name,
        os.path.join(PLUGIN_DIR, "__init__.py"),
        submodule_search_locations=[PLUGIN_DIR],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def mod():
    return _load_plugin()


@pytest.fixture
def engine(mod):
    from agent.context_compressor import ContextCompressor

    eng = mod.JevContextCompressor(model="test-model", quiet_mode=True)
    return eng


# --------------------------------------------------------------- fixtures

def make_transcript(big: int = 4000) -> list:
    """A transcript with several old tool pairs, a protected head and tail."""
    return [
        {"role": "system", "content": "You are Hermes."},
        {"role": "user", "content": "Fix the failing test in src/a.ts. Never edit src/generated."},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "c1", "type": "function",
             "function": {"name": "read_file", "arguments": '{"path": "src/a.ts"}'}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "A" * big},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "c2", "type": "function",
             "function": {"name": "grep", "arguments": '{"pattern": "TODO"}'}}]},
        {"role": "tool", "tool_call_id": "c2", "content": "B" * big},
        {"role": "assistant", "content": "Found it: the guard returns early.", "tool_calls": []},
        {"role": "user", "content": "Now add a regression test."},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "c3", "type": "function",
             "function": {"name": "terminal", "arguments": '{"command": "pytest -q"}'}}]},
        {"role": "tool", "tool_call_id": "c3", "content": "C" * big},
    ]


# ------------------------------------------------------------------- abc

def test_engine_satisfies_context_engine_abc(mod, engine):
    from agent.context_engine import ContextEngine

    assert isinstance(engine, ContextEngine)
    assert engine.name == "jev-compaction"


def test_engine_subclasses_built_in_compressor(mod, engine):
    from agent.context_compressor import ContextCompressor

    assert isinstance(engine, ContextCompressor)


def test_required_abc_methods_are_callable(mod, engine):
    engine.update_from_response(
        {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12}
    )
    assert engine.last_prompt_tokens == 10
    assert engine.should_compress(0) in (True, False)
    assert isinstance(engine.compress([{"role": "user", "content": "hi"}]), list)
    assert isinstance(engine.get_status(), dict)
    assert "jev" in engine.get_status()


# -------------------------------------------------------------- deepcopy

def test_deepcopy_succeeds(mod, engine):
    """The host deep-copies the plugin singleton per agent (#42449).

    A threading.Lock in the copy makes deepcopy raise, and the host then
    silently falls back to the built-in compressor — the plugin appears
    installed and does nothing. This is the regression guard for that.
    """
    clone = copy.deepcopy(engine)
    assert clone is not engine
    assert clone.name == "jev-compaction"
    assert isinstance(clone, mod.JevContextCompressor)


def test_deepcopy_gives_independent_lock_and_cache(mod, engine):
    clone = copy.deepcopy(engine)
    assert clone._jev_lock is not engine._jev_lock
    assert clone._jev_cache is not engine._jev_cache
    # Budget state is copied, not shared, so one agent can't move another's.
    clone._jev_stats["passes"] = 99
    assert engine._jev_stats["passes"] == 0


def test_deepcopy_copies_engine_budget_state(mod, engine):
    engine.update_from_response(
        {"prompt_tokens": 1234, "completion_tokens": 5, "total_tokens": 1239}
    )
    clone = copy.deepcopy(engine)
    assert clone.last_prompt_tokens == 1234


# ---------------------------------------------------------- candidate set

def test_protected_head_and_tail_are_never_candidates(mod):
    msgs = make_transcript()
    cands = mod.jev.iter_candidates(msgs, protect_first_n=3, protect_last_n=2,
                                    min_result_chars=200)
    ids = [c["id"] for c in cands]
    # c1 sits inside the protected head (messages 1..3 non-system).
    assert "c1" not in ids
    # c3 is inside the protected tail (last 2 messages).
    assert "c3" not in ids
    assert ids == ["c2"]


def test_explicit_zero_config_is_not_swallowed(mod):
    """Regression: `cfg.get(k) or default` rewrote a deliberate 0.

    An explicit 0 is meaningful for several knobs (protect nothing, keep only
    the note). Only a missing/unparseable value may fall back to the default.
    """
    assert mod._cfg_int({"protect_last_n": 0}, "protect_last_n", 20) == 0
    assert mod._cfg_int({"min_candidates": 0}, "min_candidates", 4) == 0
    assert mod._cfg_int({"truncate_head_chars": 0}, "truncate_head_chars", 300) == 0
    assert mod._cfg_float({"keep_threshold": 0.0}, "keep_threshold", 0.5) == 0.0
    # Unset / unparseable still falls back.
    assert mod._cfg_int({}, "protect_last_n", 20) == 20
    assert mod._cfg_int({"x": "garbage"}, "x", 7) == 7
    assert mod._cfg_float({"x": None}, "x", 1.5) == 1.5


def test_zero_protection_offers_every_pair_as_candidate(mod):
    """With protection explicitly off, every old pair is eligible."""
    msgs = make_transcript()
    ids = [c["id"] for c in mod.jev.iter_candidates(
        msgs, protect_first_n=0, protect_last_n=0, min_result_chars=200)]
    assert ids == ["c1", "c2", "c3"]


def test_skill_view_results_are_excluded(mod, engine):
    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "s1", "type": "function",
             "function": {"name": "skill_view", "arguments": '{"name": "x"}'}}]},
        {"role": "tool", "tool_call_id": "s1", "content": "S" * 4000},
        {"role": "user", "content": "go on"},
    ]
    eng = copy.deepcopy(engine)
    eng._jev_cfg = dict(eng._jev_cfg, min_candidates=1, protect_first_n=0,
                        protect_last_n=0)
    out, n = eng._jev_prepass(msgs, protect_tail_count=0)
    assert n == 0 and out is msgs


# --------------------------------------------------------- drop / trunc

def test_drop_empties_result_but_keeps_row_and_call(mod):
    """DROP removes the payload, never the row (row count is a hard contract)."""
    msgs = make_transcript()
    decisions = {"c2": "drop"}
    out, n = mod._drop_plan(msgs, decisions)
    assert n > 0
    assert len(out) == len(msgs)
    # the result row survives, with its id, but its body is gone
    result = [m for m in out if m.get("tool_call_id") == "c2"]
    assert len(result) == 1
    assert len(result[0]["content"]) < 200
    assert "re-run" in result[0]["content"]
    # ...and its call is still there, verbatim
    calls = [tc for m in out for tc in (m.get("tool_calls") or []) if tc["id"] == "c2"]
    assert len(calls) == 1
    assert calls[0]["function"]["arguments"] == '{"pattern": "TODO"}'


def test_drop_plan_preserves_row_count(mod):
    """Regression guard for the 7428 crash: a shorter list made
    ContextCompressor.compress index past the end of its own transcript."""
    msgs = make_transcript()
    for decisions in ({"c1": "drop", "c2": "drop", "c3": "drop"},
                      {"c1": "truncate", "c2": "drop"},
                      {"c1": "keep", "c2": "drop"}):
        out, _ = mod._drop_plan(msgs, decisions)
        assert len(out) == len(msgs), decisions


def test_drop_plan_is_idempotent_on_the_note(mod):
    """A second pass over an already-set-aside result reports no change."""
    msgs = make_transcript()
    once, n1 = mod._drop_plan(msgs, {"c2": "drop"})
    twice, n2 = mod._drop_plan(once, {"c2": "drop"})
    assert n1 > 0 and n2 == 0
    assert [m.get("content") for m in twice][:8] == [m.get("content") for m in once][:8]


def test_no_orphan_tool_results_after_drop(mod):
    msgs = make_transcript()
    out, _ = mod._drop_plan(msgs, {"c1": "drop", "c2": "drop"})
    call_ids = {tc["id"] for m in out for tc in (m.get("tool_calls") or [])}
    result_ids = {m.get("tool_call_id") for m in out if m.get("role") == "tool"}
    assert result_ids <= call_ids, "a result survived without its call"


def test_drop_never_removes_a_sole_call_row(mod):
    """An assistant row whose only call was dropped must not vanish: the
    compressor's index math assumes the transcript keeps its shape."""
    msgs = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "c9", "type": "function",
             "function": {"name": "read_file", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "c9", "content": "Z" * 500},
        {"role": "user", "content": "next"},
    ]
    out, n = mod._drop_plan(msgs, {"c9": "drop"})
    assert len(out) == len(msgs) == 4
    assert n == 1
    assert out[1].get("tool_calls")          # call kept
    assert len(out[2]["content"]) < 200      # payload replaced by the note


def test_drop_keeps_assistant_message_that_still_has_text(mod):
    msgs = make_transcript()
    out, _ = mod._drop_plan(msgs, {"c2": "drop", "c3": "drop"})
    texts = [m.get("content") for m in out if m.get("role") == "assistant"]
    assert "Found it: the guard returns early." in texts


def test_truncate_keeps_call_intact_and_shrinks_result(mod):
    msgs = make_transcript()
    cands = mod.jev.iter_candidates(msgs, protect_first_n=0, protect_last_n=0,
                                    min_result_chars=200)
    c2 = [c for c in cands if c["id"] == "c2"][0]
    out, n = mod._truncate_results(list(msgs), [c2], {"c2": "truncate"}, 300)
    assert n == 1
    # The call survives verbatim.
    found = [tc for m in out for tc in (m.get("tool_calls") or []) if tc["id"] == "c2"]
    assert len(found) == 1
    assert found[0]["function"]["arguments"] == '{"pattern": "TODO"}'
    # The result body is gone but the note explains how to get it back.
    result = [m for m in out if m.get("tool_call_id") == "c2"][0]
    assert len(result["content"]) < 1000
    assert "re-run" in result["content"]
    assert result["content"].startswith("B" * 10)


def test_keep_leaves_everything_alone(mod):
    msgs = make_transcript()
    out, n = mod._drop_plan(msgs, {"c1": "keep", "c2": "keep"})
    assert n == 0
    assert out == msgs


# ------------------------------------------------------------- fail open

def test_missing_key_fails_open(mod, engine, monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    monkeypatch.setattr(mod.jev, "ENV_FILE", "/nonexistent/.env")
    msgs = make_transcript()
    eng = copy.deepcopy(engine)
    eng._jev_cfg = dict(eng._jev_cfg, min_candidates=1, protect_first_n=0,
                        protect_last_n=0)
    out, n = eng._jev_prepass(msgs, protect_tail_count=0)
    assert out is msgs and n == 0


def test_network_error_fails_open(mod, engine, monkeypatch):
    def boom(*a, **k):
        raise OSError("network down")

    monkeypatch.setattr(mod.jev, "ask", boom)
    monkeypatch.setattr(mod.jev, "load_key", lambda: "fake-key")
    msgs = make_transcript()
    eng = copy.deepcopy(engine)
    eng._jev_cfg = dict(eng._jev_cfg, min_candidates=1, protect_first_n=0,
                        protect_last_n=0)
    out, n = eng._jev_prepass(msgs, protect_tail_count=0)
    assert out is msgs and n == 0
    assert eng._jev_stats["errors"] == 1


def test_below_min_candidates_skips_the_request(mod, engine, monkeypatch):
    called = []
    monkeypatch.setattr(mod.jev, "ask", lambda *a, **k: called.append(1))
    msgs = make_transcript()
    eng = copy.deepcopy(engine)
    eng._jev_cfg = dict(eng._jev_cfg, min_candidates=99)
    out, n = eng._jev_prepass(msgs, protect_tail_count=0)
    assert out is msgs and n == 0
    assert not called, "spent a request for a batch that was too thin"


def test_cooldown_suppresses_a_second_request(mod, engine, monkeypatch):
    calls = []

    def fake_ask(state, questions, **k):
        calls.append(len(questions))
        return {"answers": {n: {"noul": 0.1} for n in questions}}

    monkeypatch.setattr(mod.jev, "ask", fake_ask)
    monkeypatch.setattr(mod.jev, "load_key", lambda: "fake-key")
    msgs = make_transcript()
    eng = copy.deepcopy(engine)
    eng._jev_cfg = dict(eng._jev_cfg, min_candidates=1, protect_first_n=0,
                        protect_last_n=0, min_seconds_between_calls=60)

    eng._jev_prepass(msgs, protect_tail_count=0)
    first = len(calls)
    eng._jev_prepass(msgs, protect_tail_count=0)
    assert len(calls) == first, "cooldown did not suppress the second request"


# -------------------------------------------------------------- decisions

def test_decide_maps_probabilities_to_actions(mod):
    cands = [{"id": "a"}, {"id": "b"}, {"id": "c"}]
    answers = {
        "keep_call_a": {"noul": 0.9}, "keep_result_a": {"noul": 0.8},     # keep
        "keep_call_b": {"noul": 0.9}, "keep_result_b": {"noul": 0.1},     # truncate
        "keep_call_c": {"noul": 0.1}, "keep_result_c": {"noul": 0.1},     # drop
    }
    decisions, missing = mod.jev.decide(answers, cands, 0.5)
    assert decisions == {"a": "keep", "b": "truncate", "c": "drop"}
    assert missing == 0


def test_decide_omits_unanswered_candidates(mod):
    cands = [{"id": "a"}, {"id": "b"}]
    answers = {"keep_call_a": {"noul": 0.9}, "keep_result_a": {"noul": 0.9}}
    decisions, missing = mod.jev.decide(answers, cands, 0.5)
    assert decisions == {"a": "keep"}
    assert missing == 1
    assert "b" not in decisions, "an unanswered call must not be dropped"


def test_decide_rejects_non_numeric_answers(mod):
    cands = [{"id": "a"}]
    answers = {"keep_call_a": {"noul": "yes"}, "keep_result_a": {"noul": None}}
    decisions, missing = mod.jev.decide(answers, cands, 0.5)
    assert decisions == {} and missing == 1


# ------------------------------------------------------------- token math

def test_token_estimate_is_positive_and_monotonic(mod):
    assert mod.jev.estimate_tokens("") == 0
    small = mod.jev.estimate_tokens("hello world")
    big = mod.jev.estimate_tokens("hello world" * 100)
    assert 0 < small < big


def test_state_replaces_tool_results_with_notes(mod):
    msgs = make_transcript(big=5000)
    cands = mod.jev.iter_candidates(msgs, 0, 0, 200)
    state = mod.jev.build_state(msgs, cands, "goal", max_state_tokens=100000)
    tool_entries = [e for e in state["messages"] if e.get("role") == "tool"]
    assert tool_entries, "tool messages missing from state"
    for e in tool_entries:
        assert "chars (omitted)" in e["result"]
        assert len(e["result"]) < 60, "a full tool body leaked into the state"


def test_state_respects_the_token_budget(mod):
    msgs = make_transcript(big=8000)
    cands = mod.jev.iter_candidates(msgs, 0, 0, 200)
    state = mod.jev.build_state(msgs, cands, "goal", max_state_tokens=500)
    assert mod.jev.estimate_tokens(str(state)) <= 5000


# ------------------------------------------------------------------ live

@pytest.mark.skipif(os.environ.get("JEV_LIVE") != "1",
                    reason="set JEV_LIVE=1 to spend a real (tiny) Jev request")
def test_live_request_returns_decisions(mod, engine):
    msgs = make_transcript(big=3000)
    eng = copy.deepcopy(engine)
    eng._jev_cfg = dict(eng._jev_cfg, min_candidates=1, protect_first_n=0,
                        protect_last_n=0)
    out, n = eng._jev_prepass(msgs, protect_tail_count=0)
    assert eng._jev_stats["errors"] == 0
    assert eng._jev_stats["judged"] >= 1
    assert n >= 0

"""Contract tests for jev-memory-gate.

The important ones drive the REAL host chain
(``hermes_cli.middleware._run_execution_chain``), not a mock of it, because the
failure this plugin must never cause — returning ``None`` as a tool result —
only exists in that host code.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys

REPO = os.path.expanduser("~/.hermes/hermes-agent")
PLUGIN_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)


def _load_plugin():
    name = "jev_memory_gate_under_test"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(PLUGIN_DIR, "__init__.py"),
        submodule_search_locations=[PLUGIN_DIR])
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def fresh_gate(mod, **overrides):
    """A gate with cooldowns disabled so each test starts clean."""
    mod._GATE = None
    gate = mod.MemoryGate()
    gate.cfg = dict(mod.DEFAULT_CONFIG, min_seconds_between_calls=0,
                    error_cooldown_s=0, **overrides)
    mod._GATE = gate
    return gate


class Terminal:
    """Stands in for the real tool dispatcher."""

    def __init__(self, result="TOOL-RESULT"):
        self.calls = []
        self.result = result

    def __call__(self, args):
        self.calls.append(args)
        return self.result


def run_chain(mod, tool_name, args, terminal):
    from hermes_cli.middleware import _run_execution_chain
    return _run_execution_chain(
        "tool_execution", [mod.on_tool_execution], terminal,
        tool_name=tool_name, args=args, original_args=args,
    )


PREFERENCE = ("User prefers terse replies and mirrors language (English/Chinese); "
              "expects proactive execution rather than repeated confirmation.")
FILLER = ("Ran pytest again and it passed 26 tests. Finished task #3, committed the "
          "changes and moved on to the next item on the list.")


# ------------------------------------------------------- the critical one

def test_non_memory_tool_result_is_never_swallowed(mod):
    """A callback that returns None here would hand None back as the tool result.

    ``_run_execution_chain`` ends with ``return callback(**call_kwargs)``, so the
    middleware's return value IS the tool result. Every non-memory tool must
    therefore go through next_call.
    """
    fresh_gate(mod)
    terminal = Terminal("read_file-output")
    out = run_chain(mod, "read_file", {"path": "/tmp/x"}, terminal)
    assert out == "read_file-output", f"tool result was swallowed: {out!r}"
    assert len(terminal.calls) == 1


def test_every_other_tool_passes_through(mod):
    fresh_gate(mod)
    for tool in ("terminal", "browser_exec", "skill_view", "cronjob", "delegate_task"):
        terminal = Terminal(f"{tool}-ok")
        out = run_chain(mod, tool, {"x": 1}, terminal)
        assert out == f"{tool}-ok", f"{tool} result swallowed: {out!r}"


def test_memory_read_passes_through(mod):
    fresh_gate(mod)
    terminal = Terminal("MEMORY-CONTENT")
    out = run_chain(mod, "memory", {"action": "read", "target": "memory"}, terminal)
    assert out == "MEMORY-CONTENT"
    assert len(terminal.calls) == 1


def test_memory_remove_passes_through(mod):
    fresh_gate(mod)
    terminal = Terminal("removed")
    out = run_chain(mod, "memory", {"action": "remove", "old_text": "stale"}, terminal)
    assert out == "removed"
    assert len(terminal.calls) == 1


def test_short_entry_is_not_judged(mod, monkeypatch):
    """Tiny contents (usually trims/edits) should not cost a request."""
    fresh_gate(mod)
    called = []
    monkeypatch.setattr(mod, "judge", lambda *a, **k: called.append(1) or 0.0)
    terminal = Terminal("saved")
    out = run_chain(mod, "memory", {"action": "add", "content": "short"}, terminal)
    assert out == "saved"
    assert not called, "spent a request on a tiny entry"


def test_confident_filler_is_blocked_without_writing(mod, monkeypatch):
    fresh_gate(mod, block_below=0.25)
    monkeypatch.setattr(mod, "judge", lambda *a, **k: 0.05)
    terminal = Terminal("saved")
    out = run_chain(mod, "memory", {"action": "add", "content": FILLER}, terminal)
    assert not terminal.calls, "the write was allowed through"
    body = json.loads(out)
    assert body["saved"] is False
    assert "NOT SAVED" in body["reason"]
    assert mod._GATE.stats["blocked"] == 1


def test_a_real_preference_is_allowed_through(mod, monkeypatch):
    fresh_gate(mod, block_below=0.25)
    monkeypatch.setattr(mod, "judge", lambda *a, **k: 0.91)
    terminal = Terminal("saved")
    out = run_chain(mod, "memory", {"action": "add", "content": PREFERENCE}, terminal)
    assert out == "saved"
    assert len(terminal.calls) == 1
    assert mod._GATE.stats["passed"] == 1


def test_borderline_entry_is_allowed_through(mod, monkeypatch):
    """Only a confident rejection blocks; 0.4 is 'maybe worth it' -> let it in."""
    fresh_gate(mod, block_below=0.25)
    monkeypatch.setattr(mod, "judge", lambda *a, **k: 0.40)
    terminal = Terminal("saved")
    out = run_chain(mod, "memory", {"action": "add", "content": FILLER}, terminal)
    assert out == "saved"
    assert len(terminal.calls) == 1


def test_judge_error_fails_open(mod, monkeypatch):
    fresh_gate(mod)

    def boom(*a, **k):
        raise OSError("network down")

    monkeypatch.setattr(mod, "judge", boom)
    terminal = Terminal("saved")
    out = run_chain(mod, "memory", {"action": "add", "content": FILLER}, terminal)
    assert out == "saved", "an error must never lose a write"
    assert len(terminal.calls) == 1
    assert mod._GATE.stats["errors"] == 1


def test_no_answer_fails_open(mod, monkeypatch):
    fresh_gate(mod)
    monkeypatch.setattr(mod, "judge", lambda *a, **k: None)
    terminal = Terminal("saved")
    out = run_chain(mod, "memory", {"action": "add", "content": FILLER}, terminal)
    assert out == "saved"


def test_batch_operations_are_judged(mod, monkeypatch):
    """A `operations` array carries its own contents which must be judged too."""
    fresh_gate(mod, block_below=0.25)
    seen = []

    def fake(text, cfg):
        seen.append(text)
        return 0.05 if text == FILLER else 0.9

    monkeypatch.setattr(mod, "judge", fake)
    terminal = Terminal("saved")
    out = run_chain(mod, "memory", {
        "operations": [
            {"action": "add", "content": FILLER},
            {"action": "replace", "old_text": "x",
             "content": "User prefers metric units and ISO dates in generated documents."},
            {"action": "remove", "old_text": "gone"},
        ]
    }, terminal)
    assert not terminal.calls
    assert FILLER in seen
    assert len(seen) == 2, f"removes must not be judged (judged: {len(seen)})"


def test_missing_key_fails_open_with_real_judge(mod, monkeypatch):
    """End-to-end through the real judge(), with no credential present."""
    fresh_gate(mod)
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    monkeypatch.setattr(mod, "ENV_FILE", "/nonexistent/.env")
    terminal = Terminal("saved")
    out = run_chain(mod, "memory", {"action": "add", "content": FILLER}, terminal)
    assert out == "saved"
    assert mod._GATE.stats["skipped_no_key"] == 1


def test_the_gate_can_be_disabled(mod, monkeypatch):
    fresh_gate(mod, enabled=False)
    called = []
    monkeypatch.setattr(mod, "judge", lambda *a, **k: called.append(1) or 0.0)
    terminal = Terminal("saved")
    out = run_chain(mod, "memory", {"action": "add", "content": FILLER}, terminal)
    assert out == "saved"
    assert not called


def test_candidates_extraction(mod):
    gate = mod.MemoryGate()
    assert gate._candidates({"action": "read"}) == []
    assert gate._candidates({"action": "add", "content": "x"}) == [("add", "x")]
    assert gate._candidates({"action": "add", "content": "   "}) == []
    got = gate._candidates({"operations": [
        {"action": "add", "content": "a"},
        {"action": "remove", "old_text": "b"},
        {"action": "replace", "new_text": "c"},
    ]})
    assert got == [("batch:add", "a"), ("batch:replace", "c")]


# ------------------------------------------------------------------- live

def test_live_judgement_separates_filler_from_preference(mod):
    """Real Jev call: a genuine preference must outscore session filler."""
    if os.environ.get("JEV_LIVE") != "1":
        return "SKIP"
    gate = fresh_gate(mod)
    p_pref = mod.judge(PREFERENCE, gate.cfg)
    p_filler = mod.judge(FILLER, gate.cfg)
    print(f"      preference p={p_pref}   filler p={p_filler}")
    assert p_pref is not None and p_filler is not None
    assert p_pref > p_filler, "Jev did not rank the real preference above filler"

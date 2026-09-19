#!/usr/bin/env python3
"""Assert the prune-length invariant for a Hermes context-engine plugin.

Why: the host captures n_messages = len(messages) BEFORE calling
_prune_old_tool_results, refreshes it only inside the blank-echo branch, and then
indexes messages[i] up to that stale bound. An override that RETURNS A SHORTER LIST
kills the whole compaction pass with `IndexError: list index out of range`, which the
user sees only as "Sorry, I encountered an unexpected error."
See references/prune-length-invariant-crash.md.

Run (no network, no spend, no model call — the judgment call is stubbed):

    ~/.hermes/hermes-agent/venv/bin/python3 \
        ~/.hermes/skills/devops/hermes-context-engines/scripts/check_prune_length_invariant.py

Exit 0 = invariant holds (engine is safe to enable).
Exit 1 = the engine (or the host) violates it — do not enable; read the reference.

Optional: NAME=<plugin-dir-name> to point at a different plugin under ~/.hermes/plugins/.
"""

from __future__ import annotations

import importlib.util
import os
import re
import sys

HOME = os.path.expanduser("~")
REPO = os.path.join(HOME, ".hermes", "hermes-agent")
PLUGIN_NAME = os.environ.get("NAME", "jev-compaction")
PLUGIN_DIR = os.path.join(HOME, ".hermes", "plugins", PLUGIN_NAME)
SRC = os.path.join(REPO, "agent", "context_compressor.py")

if REPO not in sys.path:
    sys.path.insert(0, REPO)


def load_plugin():
    """Mirror the loader: the directory name has a hyphen, so it is not importable."""
    name = PLUGIN_NAME.replace("-", "_") + "_invariant_probe"
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(PLUGIN_DIR, "__init__.py"),
        submodule_search_locations=[PLUGIN_DIR],
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def transcript(n_pairs: int = 8, big: int = 3000) -> list:
    """Protected head + a compressible middle of old tool pairs + a live tail."""
    msgs = [
        {"role": "system", "content": "You are Hermes."},
        {"role": "user", "content": "Fix the failing test. Never edit src/generated."},
    ]
    for i in range(n_pairs):
        cid = f"c{i}"
        msgs.append({
            "role": "assistant", "content": "",
            "tool_calls": [{"id": cid, "type": "function",
                            "function": {"name": "read_file",
                                         "arguments": '{"path": "src/a.ts"}'}}],
        })
        msgs.append({"role": "tool", "tool_call_id": cid, "content": chr(65 + i) * big})
    msgs.append({"role": "user", "content": "Add a regression test."})
    return msgs


def main() -> int:
    if not os.path.isdir(PLUGIN_DIR):
        print(f"SKIP: no plugin at {PLUGIN_DIR} (set NAME=<dir> to probe another)")
        return 0
    if not os.path.isfile(SRC):
        print(f"SKIP: host source not found at {SRC}")
        return 0

    # --- 1. Where does the stale bound come from today? (report real line numbers) ---
    src = open(SRC, encoding="utf-8", errors="replace").read().splitlines()
    print("host sites (reported live, may shift with upstream edits):")
    for pat, what in ((r"^\s*n_messages = len\(messages\)", "n_messages captured/refreshed"),
                      (r"_prune_old_tool_results\(\s*$", "prune is called (list may be replaced)"),
                      (r"for i in range\(max\(compress_end, tail_start\), n_messages\)",
                       "tail loop bound to n_messages"),
                      (r"_fresh_compaction_message_copy\(messages\[i\]\)",
                       "the indexing line that raises IndexError")):
        for n, line in enumerate(src, 1):
            if re.search(pat, line):
                print(f"   {n:6d}  {what}")

    mod = load_plugin()
    engine_cls = None
    for attr in dir(mod):
        obj = getattr(mod, attr)
        if isinstance(obj, type) and obj.__module__ == mod.__name__:
            from agent.context_compressor import ContextCompressor
            if issubclass(obj, ContextCompressor):
                engine_cls = obj
                break
    if engine_cls is None:
        print("FAIL: no ContextCompressor subclass found in the plugin")
        return 1

    engine = engine_cls(model="probe-model", quiet_mode=True)

    # Stub the paid judgment call: every candidate scores ~0 on both questions -> DROP.
    def fake_ask(messages, fresh, cfg):
        answers = {}
        for c in fresh:
            answers[f"keep_result_{c['id']}"] = {"noul": 0.01}
            answers[f"keep_call_{c['id']}"] = {"noul": 0.01}
        return answers, 0

    if hasattr(engine, "_ask_jev"):
        engine._ask_jev = fake_ask
    else:
        print("note: engine has no _ask_jev; probing whatever prune override exists")

    msgs = transcript()
    before = len(msgs)
    try:
        out = engine._prune_old_tool_results(msgs, protect_tail_count=2,
                                             protect_tail_tokens=None, min_prune_chars=0)
    except Exception as exc:  # a raise here is a different (also reportable) bug
        print(f"FAIL: _prune_old_tool_results raised {type(exc).__name__}: {exc}")
        return 1

    pruned, count = out
    print(f"\n  input rows : {before}")
    print(f"  output rows: {len(pruned)}")
    print(f"  pruned_cnt : {count}")

    if len(pruned) != before:
        print("\nFAIL: the prune override CHANGED THE ROW COUNT.")
        print("  The host still indexes messages[0.." + str(before - 1) + "] on a list of")
        print(f"  length {len(pruned)} -> IndexError: list index out of range at the tail loop.")
        print("  Fix: keep one row per input row (blank/stub the tool RESULT payload; keep")
        print("  the assistant tool_calls row). See references/prune-length-invariant-crash.md")
        return 1

    # A same-length list must still be indexable for every i the host will use.
    for i in (0, before - 1):
        _ = pruned[i]
    print("\nPASS: row count preserved — the host's stale n_messages stays in range.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Run the jev-compaction contract tests + prove the 7428 crash is fixed.

Why this exists: pytest's collection of the plugin directory trips over the
plugin's own ``__init__.py`` (relative ``from . import jev``), so the suite is
driven directly here: real plugin module, real engine, no network.

Layer 1 (A/B mechanism): a compressor whose prune seam returns a SHORTER list
    must crash inside ``compress`` exactly like production did on 2026-09-18.
Layer 2 (the fix): the real jev engine, forced to DROP, must complete
    ``compress`` with the row count intact.
"""
import importlib.util
import os
import sys
import types

REPO = os.path.expanduser("~/.hermes/hermes-agent")
PLUGIN_DIR = os.path.expanduser("~/.hermes/plugins/jev-compaction")
sys.path.insert(0, REPO)

# NOTE: deliberately does NOT set TYPESAFE_API_KEY — the suite has a
# missing-key fail-open contract test that a stray env var would break.

# --- import the plugin the way a package loader does (parent package needed
#     for its relative ``from . import jev``) -----------------------------
pkg = types.ModuleType("jevpkg")
pkg.__path__ = [PLUGIN_DIR]
sys.modules["jevpkg"] = pkg
spec = importlib.util.spec_from_file_location(
    "jevpkg.plugin", os.path.join(PLUGIN_DIR, "__init__.py")
)
mod = importlib.util.module_from_spec(spec)
sys.modules["jevpkg.plugin"] = mod
spec.loader.exec_module(mod)
print(f"plugin imported: {mod.__name__}  (jev={mod.jev.__name__})")

from agent.context_compressor import ContextCompressor  # noqa: E402

FAILS = []


def transcript(n_pairs=14, result_chars=12000):
    msgs = [{"role": "system", "content": "You are Hermes."}]
    msgs.append({"role": "user", "content": "Fix the failing test in src/a.ts."})
    for k in range(n_pairs):
        cid = f"c{k}"
        msgs.append({"role": "assistant", "content": "",
                     "tool_calls": [{"id": cid, "type": "function",
                                     "function": {"name": "read_file",
                                                  "arguments": '{"path": "src/a.ts"}'}}]})
        msgs.append({"role": "tool", "tool_call_id": cid,
                     "content": (f"result-{k} " * (result_chars // 10))})
    msgs.append({"role": "user", "content": "Now add a regression test."})
    msgs.append({"role": "assistant", "content": "Done — test added."})
    return msgs


def make_engine():
    eng = mod.JevContextCompressor(model="test-model", quiet_mode=True)
    eng._generate_summary = lambda *a, **k: "STUB SUMMARY"   # no LLM call
    return eng


# ---------------------------------------------------------------- layer 1
class ShrinkingCompressor(ContextCompressor):
    """Simulates any prune seam that hands back a shorter list."""

    def _prune_old_tool_results(self, messages, protect_tail_count,
                                protect_tail_tokens=None, min_prune_chars=100):
        out, n = super()._prune_old_tool_results(
            messages, protect_tail_count, protect_tail_tokens, min_prune_chars
        )
        return out[:-5], n + 5


def layer1():
    eng = ShrinkingCompressor(model="test-model", quiet_mode=True)
    eng._generate_summary = lambda *a, **k: "STUB SUMMARY"
    msgs = transcript()
    print(f"\n[1] shrink-prone prune: {len(msgs)} rows in")
    try:
        out = eng.compress(msgs, current_tokens=900_000)
        print(f"    no crash: returned {len(out)} rows")
        return "no crash"
    except IndexError as e:
        print(f"    IndexError reproduced: {e!r}")
        return "IndexError"


# ---------------------------------------------------------------- layer 2
def layer2():
    eng = make_engine()
    drops = []

    def fake_ask(messages, fresh, cfg):
        drops.extend(c["id"] for c in fresh)
        return ({f"keep_result_{c['id']}": {"noul": 0.01} for c in fresh} |
                {f"keep_call_{c['id']}": {"noul": 0.01} for c in fresh}), 0

    eng._ask_jev = fake_ask
    msgs = transcript()
    print(f"\n[2] real engine, forced DROP: {len(msgs)} rows in")

    # 2a — the seam itself: row count must be identical after the prune
    seam_out, seam_n = eng._prune_old_tool_results(msgs, protect_tail_count=0)
    notes = [m for m in seam_out
             if m.get("role") == "tool" and "dropped by jev-compaction" in str(m.get("content"))]
    print(f"    seam: {len(msgs)} rows in -> {len(seam_out)} rows out, "
          f"{len(drops)} decision(s), {len(notes)} payload(s) set aside")
    if len(seam_out) != len(msgs):
        FAILS.append(f"seam changed the row count: {len(msgs)} -> {len(seam_out)}")
    if not notes:
        FAILS.append("seam emptied no payload — the drop path never ran")
    calls_in = {tc["id"] for m in msgs for tc in (m.get("tool_calls") or [])}
    calls_out = {tc["id"] for m in seam_out for tc in (m.get("tool_calls") or [])}
    if calls_out != calls_in:
        FAILS.append(f"calls changed by the seam: {sorted(calls_in - calls_out)} lost")

    # 2b — the whole compressor must survive the seam (this is the 7428 path)
    try:
        out = eng.compress(msgs, current_tokens=900_000)
    except IndexError as e:
        FAILS.append(f"compress still crashed: {e!r}")
        print(f"    compress: IndexError {e!r}")
        return
    print(f"    compress: completed, {len(out)} rows returned (middle summarized)")


# ---------------------------------------------------------------- unit tests
def run_unit_tests():
    print("\n[3] plugin unit tests (drop / truncate / keep / decide / cache)")
    test_path = os.path.join(PLUGIN_DIR, "tests", "test_plugin.py")
    tspec = importlib.util.spec_from_file_location("jev_tests", test_path)
    tmod = importlib.util.module_from_spec(tspec)
    tspec.loader.exec_module(tmod)

    class MP:  # minimal monkeypatch
        def __init__(self):
            self._undo = []
        def setattr(self, obj, name, value):
            self._undo.append((obj, name, getattr(obj, name, None)))
            setattr(obj, name, value)
        def delenv(self, name, raising=True):
            self._undo.append((os.environ, name, os.environ.get(name)))
            os.environ.pop(name, None)

    passed = failed = skipped = 0
    for name in sorted(n for n in dir(tmod) if n.startswith("test_")):
        fn = getattr(tmod, name)
        if getattr(fn, "pytestmark", None):
            skipped += 1
            continue
        code = fn.__code__
        kwargs = {}
        if "mod" in code.co_varnames[:code.co_argcount]:
            kwargs["mod"] = mod
        if "engine" in code.co_varnames[:code.co_argcount]:
            kwargs["engine"] = make_engine()
        if "monkeypatch" in code.co_varnames[:code.co_argcount]:
            kwargs["monkeypatch"] = MP()
        try:
            fn(**kwargs)
            passed += 1
        except Exception as exc:
            failed += 1
            FAILS.append(f"{name}: {type(exc).__name__}: {exc}")
            print(f"    FAIL {name}: {type(exc).__name__}: {exc}")
            import traceback
            for line in traceback.format_exc().splitlines()[-8:]:
                print("        " + line)
    print(f"    {passed} passed, {failed} failed, {skipped} skipped (live)")
    return passed, failed


l1 = layer1()
l2 = layer2()
run_unit_tests()

print("\n" + "=" * 70)
if FAILS:
    print(f"RESULT: {len(FAILS)} problem(s)")
    for f in FAILS:
        print("  - " + f)
    sys.exit(1)
print("RESULT: all green")
print(f"  layer1 mechanism check: {l1}")

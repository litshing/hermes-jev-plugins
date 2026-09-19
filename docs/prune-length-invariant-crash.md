# The prune-length invariant — how a plugin engine crashed compaction

Read this alongside `references/plugin-engine-corrections.md`. It corrects two things
the rest of this skill still says:

> **SKILL.md status block is stale — do not repeat it to the user.** It reads
> *"Status: designed, NOT built… do not describe it as working."* An engine now
> exists (`~/.hermes/plugins/jev-compaction/`), is enabled, `context.engine` points
> at it, and it **has run live** (one pass logged `80 decision(s) — 80 dropped`).
> Treat it as shipped-but-buggy, not hypothetical. One-line rollback:
> `hermes config set context.engine compressor` (`context.engine` is resolved at
> agent init, so a gateway restart is what makes the change real).
> `plugin-engine-corrections.md` §2 ("not executed under a live agent") is stale for
> the same reason.
>
> *Why this is a reference file and not a SKILL.md edit:* the curator
> read-before-write guard needs a **content-returning** `skill_view` in the same
> review turn. SKILL.md had already been loaded in that turn, so every re-view
> returned `dedup: true, content_returned: false` and the patch was refused. Fold
> these corrections into SKILL.md from a fresh session.

## The failure, verbatim (2026-09-18, gateway, live session)

User-visible symptom — the generic gateway message, no detail:

```
Sorry, I encountered an unexpected error.
Try again or use /reset to start a fresh session.
```

The real traceback (`~/.hermes/logs/errors.log`, mirrored in `gateway.error.log`):

```
File ".../gateway/run.py", line 19655, in _handle_message_with_agent
File ".../agent/turn_context.py", line 1004, in build_turn_context
  messages, active_system_prompt = agent._compress_context(...)
File ".../run_agent.py", line 8067, in _compress_context
  result = run_compress_context_with_progress_timeout(...)
File ".../agent/conversation_compression.py", line 3032, in compress_context
  compressed = compress_fn(messages, **compress_kwargs)
File ".../agent/context_compressor.py", line 7428, in compress
  msg = _fresh_compaction_message_copy(messages[i])
IndexError: list index out of range
```

Same-window log lines that identify the trigger:

```
22:01:08  hermes_plugins.jev_compaction: jev-compaction: 80 decision(s)
          — 80 dropped, 0 truncated (threshold 0.50, cache 40)
22:04:44  Session hygiene: 414 messages, ~392,812 tokens — auto-compressing
22:01:41 / 22:02:51 / 22:03:25 / 22:05:26   context compression … IndexError (repeats every turn)
22:04:51  Invalidated run generation → session_reset
```

## The mechanism (line-by-line, in `agent/context_compressor.py`)

| Line | What happens |
|---|---|
| `6949` | `n_messages = len(messages)` — captured **before** the prune |
| `6990` | `messages, pruned_count = self._prune_old_tool_results(...)` — **rebinds** `messages`; a subclass override may return a *different length* |
| `7001–7007` | `n_messages = len(messages)` — refreshed **only** inside `if blank_echo_indices:` |
| `7011` / `7015` | `compress_start` / `compress_end` — recomputed *after* the prune, so they stay consistent |
| `7423` | `for i in range(max(compress_end, tail_start), n_messages):` — stale upper bound |
| `7428` | `msg = _fresh_compaction_message_copy(messages[i])` — **IndexError** |
| `7216` | `tail_msgs = n_messages - tail_start` — same stale value silently corrupts telemetry |

The built-in is immune because its own `_prune_old_tool_results` (`:3399`) starts with
`result = [m.copy() for m in messages]` — **same length, always**. The stale bound is
latent until an override changes the length; the plugin was the first engine to do so,
which is why the crash had no precedent in the logs.

**Rule: `_prune_old_tool_results` may shrink message *content*, never the message
*list*.** The host's index arithmetic, `tail_start` bookkeeping and telemetry all
assume row-for-row correspondence with what it passed in.

## Fix options

1. **Plugin-side (preferred — no core edits).** After the drop plan, keep the row count
   identical: for a `drop` decision blank/replace the **tool result payload** with a
   short stub and leave the assistant `tool_calls` row in place. The savings come from
   the payload, not the row; a ~10-token stub per dropped call is noise next to the
   kilobytes it removes. Pairing (`tool_call` ↔ `tool_call_id`) is preserved for free,
   so the "no orphan tool result" invariant also stops being something you have to
   hand-maintain.
2. **Core-side (correct, but not a plugin's business).** Refresh `n_messages =
   len(messages)` unconditionally right after the `:6990` call. If you take this route,
   treat it as a core bug fix with its own repro/PR, and do **not** let the plugin
   depend on it.

## Reproduction (deterministic, no network, no spend)

Minimal core-only repro of the latent bug — proves the mechanism without the plugin:

```python
class Shrinking(ContextCompressor):
    def _prune_old_tool_results(self, messages, protect_tail_count,
                                protect_tail_tokens=None, min_prune_chars=0):
        return messages[:-10], 10          # any shorter list
```

Full plugin-path repro (exercises the real drop path):

1. Load the plugin the way the loader does (hyphenated dir):
   `importlib.util.spec_from_file_location(name, "<plugin>/__init__.py",
   submodule_search_locations=[plugin_dir])`.
2. `engine = mod.JevContextCompressor(model="test-model", quiet_mode=True)`.
3. Stub the judgment call so nothing hits the network:
   `engine._ask_jev = lambda msgs, fresh, cfg: ({f"keep_result_{c['id']}": {"noul": 0.01},
   f"keep_call_{c['id']}": {"noul": 0.01} for c in fresh}, 0)` → `jev.decide` maps every
   candidate to `DROP`.
4. Stub the summariser (`engine._generate_summary = lambda *a, **k: "STUB"`) so the
   pass does not call a model.
5. Call `engine.compress(transcript)` on a transcript with a protected head, a
   compressible middle and several old tool pairs → `IndexError` at `:7428`.

Also worth unit-testing directly: **`len(out) == len(messages)` after
`_prune_old_tool_results`**, for any override. That one assertion would have caught
this before production did.

## Diagnosing this class of failure ("unexpected error", no detail)

The user-visible message carries nothing. The trail is always in the logs:

```bash
# which logs exist and which one the gateway is actively writing
~/.hermes/logs/{errors.log,gateway.error.log,gateway.log,agent.log}
```

Procedure that worked: find the last `Traceback (most recent call last):` **before**
the exception line (scan backwards from the `IndexError`/`Exception` line, not forwards
from the file start), then read the frames **outward→inward** — the innermost frame is
where it broke, the outermost tells you *which entry point* (gateway turn vs cron vs
hygiene pass vs `/compress`). Then map the frame's `file:line` against the live source
in `~/.hermes/hermes-agent/` to see the actual expression.

Corroborate with the same-window plugin log lines (`hermes_plugins.*`) — this crash was
only attributable because `jev-compaction` logged its decision count one minute earler.

⚠️ **Never answer "did this happen before?" from an absent log.** The pre-migration
the legacy tree and its runtime logs are gone from this machine (no `.pm2`, no
the parked legacy tree, no claw logs), so prior occurrences **cannot** be
confirmed or denied. Say that instead of inferring a shared cause from a shared symptom.

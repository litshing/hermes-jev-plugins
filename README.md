# hermes-jev-plugins

Two [Hermes Agent](https://hermes-agent.nousresearch.com/docs) plugins that put a
**small, cheap judgement call** in front of two expensive things: the context
window, and permanent memory.

Both talk to TypeSafe's **System One** endpoint (model `jev-latest`) — an API for
asking many small independent probability questions about one state, instead of
paying for one big LLM call. Protocol reference: [`docs/jev-system-one-protocol.md`](docs/jev-system-one-protocol.md).

| Plugin | What it judges | Effect |
|---|---|---|
| [`jev-compaction`](jev-compaction/) | each old tool call, individually | prune by *need* instead of by *size* |
| [`jev-memory-gate`](jev-memory-gate/) | each candidate memory entry | refuse to store filler that would tax every future turn |

Both are **conservative and fail-open**: a missing key, a timeout, a bad response
or an unanswered question degrades to *do nothing* and the caller proceeds
exactly as the built-in behaviour would.

---

## jev-compaction

A **context engine** that replaces the built-in size-based pruning. It subclasses
the built-in `ContextCompressor`, so it keeps the deterministic dedup pass, the
prompt-cache hysteresis and the session-DB persistence — it only changes *which*
old tool output survives.

Per candidate tool call it asks two questions:

```
keep_call_<id>    — is the fact that this call happened still relevant?
keep_result_<id>  — are its contents still needed verbatim, i.e. would re-running not serve?
```

then, at `keep_threshold`:

```
keep_result >= t   → keep call + result verbatim
keep_call   >= t   → keep the call, shrink the result to a head + a re-run note
otherwise          → drop the pair
```

**Measured**, on a 32-message / 13-tool-pair transcript against the same
size-based baseline:

| | tool pairs | est. tokens | saved |
|---|---|---|---|
| original | 13 | 13,156 | — |
| size-based only | 13 | 11,361 | 1,795 |
| decision-based | 5 | 5,722 | 7,434 (**4.1×**) |

Text, user turns and assistant reasoning are never summarised — only tool
payloads are touched. Cost is a fraction of a cent per pass (see the protocol doc
for the measured table), with a minimum-batch gate, a cooldown and an
input-keyed memo so a per-turn hook cannot re-bill for identical inputs.

> ⚠️ **Row count is a hard contract.** Hermes captures `n_messages = len(messages)`
> *before* calling `prune_old_tool_results` and indexes with that stale bound, so a
> prune seam that returns a **shorter list** kills the whole compaction pass with
> `IndexError: list index out of range` — surfacing to the user as
> *"Sorry, I encountered an unexpected error."* A DROP here therefore **empties the
> payload and keeps the row**. Postmortem:
> [`docs/prune-length-invariant-crash.md`](docs/prune-length-invariant-crash.md);
> checker: [`scripts/check_prune_length_invariant.py`](scripts/check_prune_length_invariant.py).

## jev-memory-gate

A judgement in front of the `memory` tool. Memory is injected into *every* future
conversation, so a wasted entry is a permanent tax — the gate asks whether a
candidate entry is durable, non-obvious and steering-reducing, and declines the
write when it is confidently filler. Reads and every other tool pass straight
through; batch writes are judged entry by entry.

One `noul` question, answered against a state built from the entry text alone
(no transcript is sent):

```
worth_remembering — noul: high = "not worth carrying across sessions"
   false criterion: durable, non-obvious, steering-reducing, user-specific
   true  criterion: task progress, completed-work logs, session outcomes,
                    restated context, ephemeral state, raw data dumps,
                    anything trivially rediscoverable
```

### Thresholds and cost control (defaults, from the source)

| Knob | Default | Why |
|---|---|---|
| `block_below` | **0.25** | Decline only when P(worth keeping) < 0.25. Deliberately low: blocking a real preference costs more than storing a filler entry. |
| `error_cooldown_s` | **120** | After a failure, no further calls for 2 minutes. |
| short entries | never judged | Tiny contents (trims, edits) don't cost a request. |
| memo | keyed on entry text | A repeated candidate is never re-billed. |
| counters | `judged / passed / blocked / skipped_no_key / skipped_cooldown / errors` | Logged at INFO when `log_decisions` is on. |

**Cost.** One question about one entry — the smallest request this API supports.
For scale, the protocol doc measures a *larger* case (8 questions on a small
state) at ~1,540 input tokens ≈ **$0.00007**; the gate's payload is bounded by
that, and the knobs above exist to keep it there rather than to measure a
per-entry average (this repo does not claim a measured per-entry figure).

### Permissions — read this or the gate does nothing

Declining a write means **short-circuiting the `memory` tool call** (returning a
tool result instead of calling `next_call`). The host grants that only with the
**`tools.override`** capability, and *everything defaults OFF*:

```bash
hermes plugins capabilities jev-memory-gate     # declared vs granted, effective state
```

Without the grant the plugin loads and judges, but the host refuses every decline
— you will see this in the logs and the gate **fails open silently**:

```
capability=tools.override decision=deny checked_by=plugin_capability_granted evidence=not granted
```

Grant it either way:

```bash
hermes plugins update jev-memory-gate     # consent flow, interactive Y/n
```

```yaml
plugins:
  entries:
    jev-memory-gate:
      granted_capabilities: [tools.override]
```

(The deprecated `allow_tool_override: true` opens the same gate.) Capabilities
are not a sandbox — in-process plugins are trusted code; this is a consent and
audit trail for host API surfaces, and it is the one thing standing between a
judgement and an actual block.

---

## Install

```bash
git clone https://github.com/litshing/hermes-jev-plugins.git
cp -R hermes-jev-plugins/jev-compaction hermes-jev-plugins/jev-memory-gate ~/.hermes/plugins/
```

Then activate — **discovery is not activation**:

```bash
hermes plugins enable jev-compaction
hermes plugins enable jev-memory-gate
hermes gateway restart
```

Select the engine in `~/.hermes/config.yaml`:

```yaml
context:
  engine: jev-compaction     # must be exactly this name, or it silently falls back
```

Provide the key (both plugins read `TYPESAFE_API_KEY`, first from the environment,
then from the `TYPESAFE_API_KEY=` line in `~/.hermes/.env`):

```bash
hermes plugins env jev-compaction     # or set TYPESAFE_API_KEY directly
```

## Verify

```bash
# contract tests — run from a NEUTRAL cwd: the hyphen in `jev-compaction` makes
# pytest try to import the plugin package and collection fails from inside it
cd /tmp
~/.hermes/hermes-agent/venv/bin/python3 -m pytest \
    ~/.hermes/plugins/jev-compaction/tests/test_plugin.py -q

# no-network, no-spend, no-model proof that the prune-length invariant holds
python3 scripts/check_prune_length_invariant.py

# two-layer A/B proof of the 2026-09-18 crash: layer 1 reproduces it with a
# shorter-list prune seam, layer 2 shows the real engine completing with the
# row count intact
python3 jev-compaction/tests/ab_prune_seam.py
```

The suite is 29 tests; the one that is skipped needs a live key and network.

## Layout

```
jev-compaction/       plugin.yaml · __init__.py · jev.py · tests/
jev-memory-gate/      plugin.yaml · __init__.py · tests/
scripts/              check_prune_length_invariant.py
docs/                 jev-system-one-protocol.md · prune-length-invariant-crash.md
gate.sh               secret scan — refuses to publish key-shaped strings
```

## License

MIT — see [LICENSE](LICENSE).

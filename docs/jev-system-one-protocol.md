# TypeSafe System One / Jev — verified request shape

A cheap "ask many small independent probability questions about one state" API.
Useful whenever a plugin needs N small judgements on a body of text and a full
LLM call would be wasteful. Verified live (response reported `model: jev-1.13.0`
for the alias `jev-latest`).

This is the same protocol used in production by `~/antiques/typesafe_gate.py`
(judging auction lots). The reference implementation of the request/response
plumbing lives there — read it before writing a new client.

## Endpoint

```
POST https://api.typesafe.ai/v1/systemone
Authorization: Bearer $TYPESAFE_API_KEY
Content-Type: application/json
```

Key resolution order that works well: `os.environ["TYPESAFE_API_KEY"]` first,
then fall back to parsing the `TYPESAFE_API_KEY=` line out of
`~/.hermes/.env`. Never log or echo the key.

## Request body

```json
{
  "model": "jev-latest",
  "state": { "...arbitrary JSON describing the situation..." },
  "questions": {
    "<question_name>": { "type": "noul",  "instructions": "...", "criteria": {"true": "...", "false": "..."} },
    "<other_name>":    { "type": "score", "instructions": "...", "criteria": ["0 label", "1 label", "..."] },
    "<third_name>":    { "type": "choice","instructions": "...", "criteria": {"key_a": "...", "key_b": "..."} }
  }
}
```

- `state` is free-form — a flat dict of text facts, or a structured object. Keep
  it text-only if the point is to avoid sending expensive payloads.
- Question names are yours; they are the keys you read answers back by.
- `instructions` should state the *decision being made* and what the question is
  being used for, not just the literal question. `criteria` pins the semantics of
  each answer so repeated calls stay calibrated.
- Ask about **one thing per question**, and keep the number of questions per
  request modest — the whole `state` is re-sent with the request.

## Response shape

```json
{
  "model": "jev-1.13.0",
  "answers": {
    "<question_name>": { "noul": 0.17 },
    "<other_name>":    { "score": 3 },
    "<third_name>":    { "choice": "porcelain" }
  },
  "usage": { "input_tokens": 1540, "output_tokens": 156 }
}
```

Accessors:

- `noul` → a probability in `[0, 1]`. The name is the label for the *negative*
  pole, so **higher `noul` = more likely the answer is "no / not this"**.
  Confirm the polarity against a working call before trusting it in a threshold —
  it is easy to invert.
- `score` → numeric, matching the `criteria` list indices.
- `choice` → one of the `criteria` keys.

Validate defensively: reject non-numeric `noul`, missing answer objects, and
malformed JSON. A missing answer must mean **"leave unchanged"**, never a
default action — silence is not a decision to delete something.

## Concurrency and batching

Questions are independent, so split a large set into multiple **concurrent**
requests; the same `state` is resent with each. Answers are then merged. Cap the
fan-out so one pathological state cannot spawn unbounded requests.

## Cost (measured)

| Scenario | Input tokens | Approx. cost |
|---|---|---|
| 8 questions on a small state | ~1,540 | ~$0.00007 |
| A real compaction pass (~13k state) | ~13,000 | ~$0.0005 |

Tiny, but not free — at one request per turn in a long tool loop it adds up.
Always build in a minimum-batch gate, a cooldown, and an in-process memo keyed on
the inputs (see the main SKILL.md).

## Worked example — conversation pruning

The strongest pattern found: instead of asking a model to *summarize* old
context (lossy), ask per tool call whether it is still needed, then delete or
truncate rather than rewrite.

Two `noul` questions per candidate call:

```
keep_call_<id>    — "Earlier the assistant called <tool> with input <args>.
                     Is retaining that this call happened still relevant?"
keep_result_<id>  — "That call returned <N> characters. Are its contents still
                     needed verbatim, such that re-running would not serve?"
```

Then, with `keep_threshold = 0.5`:

```
keep_result >= threshold              → keep call + result verbatim
else keep_call >= threshold           → keep the call, truncate the result
                                        to a head + a note saying how to re-run
else                                  → drop the call and its result together
```

Measured on a 32-message / 13-tool-pair transcript against the same
size-based baseline:

| | pairs | est. tokens | saved |
|---|---|---|---|
| original | 13 | 13,156 | — |
| size-based only | 13 | 11,361 | 1,795 |
| decision-based | 5 | 5,722 | 7,434 (**4.1×**) |

The decisions were the interesting part, and they were *right*: the pytest error
text and the latest file read were kept verbatim, the skill body was kept, the
irrelevant greps and six lint dumps were dropped entirely. Constraints and
conclusions in user/assistant *text* were never touched.

Keep state cheap: replace each tool result with a note like
`"ok, 4820 chars (omitted)"` — the contents are what you are asking *about*, not
what you show. Include tool inputs and message texts; they carry the intent.

## Pitfalls

- **Polarity.** Verify which direction `noul` points with a known-answer probe
  before wiring a threshold to it.
- **Silence ≠ delete.** An unanswered or unparseable question must fall through
  to "keep", never to the destructive branch.
- **Fail open at the boundary.** Wrap the whole call so a missing key, network
  error, HTTP error, or bad JSON degrades to "no decisions" and the caller
  proceeds unchanged. Never let this raise into a compaction or cron path.
- **Cost gates are not optional.** Minimum batch, cooldown, and an input-keyed
  memo. Without them a per-turn hook re-bills for identical inputs.
- **Don't send what you don't need.** Text-only and note-ified payloads keep the
  cost proportional to the decision, not to the raw data size.

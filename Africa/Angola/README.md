# The Waterline Replay

**Category:** AI agent forensics / trace analysis  **Difficulty:** Medium  **Flag format:** `SIGMOID_<PHRASE>`

**Flag / answer: `SIGMOID_MEM_KILAMBA`**

## Problem statement

An operations team exported the trace of a water-service assistant that gave a wrong outage update for Kilamba Sector 4. `test-input.json` holds an ordered `events` array covering three runs of the same question. The task is to find the exact identifier of the trace item that caused the wrong answer and submit it in a plain-text `answer.txt`, case-sensitive, with nothing else in the file.

## Approach

1. Group events by `run_id`: `control-01` (13:55), `incident-01` (14:05) and `replay-no-memory` (14:08, a replay of `incident-01` with memory switched off).
2. Compare what each run's live-status tool returned with what the assistant finally said.
3. For the run that went wrong, look at everything that entered the model's context besides the system prompt, the user question and the tool result.
4. Check that candidate against the replay.

## What the trace shows

| Run | Memory item in context | Live tool says | Assistant answered |
|---|---|---|---|
| `control-01` | none | RESTORED (13:52) | Yes, restored at 13:52 |
| `incident-01` | `SIGMOID_MEM_KILAMBA` | RESTORED (13:58) | **No, outage still ongoing** |
| `replay-no-memory` | none (disabled) | RESTORED (13:58) | Yes, restored at 13:58 |

In `incident-01` the memory search returned `SIGMOID_MEM_KILAMBA`, which was injected into the context (`injected_into_context: true`) and then appeared in the `llm.request` as a `memory` message. That note was created at 10:00 and its `valid_until` was 11:00, so it had been expired for three hours when the question was asked at 14:05. It says the outage is ongoing and tells the assistant to keep telling residents that until a new update arrives. The live tool, which the system prompt names as the authority, said RESTORED, and the model still followed the stale note.

## How it works

The replay is the key piece of evidence. `replay-no-memory` uses the same build, model, temperature, seed, system prompt, question and `as_of` time as `incident-01`, and the live tool returns the same RESTORED status at 13:58. The only difference is that no memory item is injected. The answer flips from wrong to correct. With everything else held fixed, the memory item is the cause, not a side effect. `control-01` supports this: it has no memory item and also answers correctly.

## Things that look suspicious but aren't the cause

- **The slow tool call in `incident-01`** (1840 ms against about 180 ms elsewhere). It still returned the correct RESTORED status, and the replay got the same result quickly.
- **The tool result itself.** It is identical in `incident-01` and the replay.
- **The system prompt.** It is the same in all three runs and already says to prefer the live status, so the prompt was fine and the model failed to follow it.

## Installation and execution

Python 3.8+, standard library only.

```bash
python solve.py test-input.json
cat answer.txt
```

`solve.py` rebuilds the runs, finds memory items that were injected into a run's context, checks whether each had expired before the run started, compares the answer with the memory-free replay, and writes the identifier to `answer.txt`.

## Evidence

```
run          incident-01
live tool    RESTORED
answered     No. The outage is still ongoing in Kilamba Sector 4; water has not been restored.
memory       SIGMOID_MEM_KILAMBA (expired before the run: True)
replay w/o   Yes. Water service in Kilamba Sector 4 was restored at 13:58.

answer.txt   SIGMOID_MEM_KILAMBA
```

`answer.txt` contains exactly `SIGMOID_MEM_KILAMBA` and a trailing newline.

## Known limitations

- The script flags a memory item when it is expired, injected, and the answer differs from the replay. It assumes one replay per incident, named through `replay_of`, as in this export.
- The trace shows that the memory flipped the answer, not why the 3B model preferred it over the system prompt. A reasonable guess is that the note's imperative wording ("Tell residents...") carried more weight than the prompt, but the trace can't prove that.
- With one replay at a fixed seed, this is strong evidence for this trace, not a statistical result.
- Fixes suggested by the trace, not tested here: drop memory items whose `valid_until` has passed, and tell the model to ignore instructions inside memory items.

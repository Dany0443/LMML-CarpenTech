# The Drying Floor at Karongi

## Task

Fix the LangGraph drying-floor state machine so every batch is inspected,
dried only when necessary, approved when its moisture is between 10.5 and
12.5, and graded after approval. The evaluator releases the flag
`SIGMOID_CONVERGENCE` only when all dataset batches finish within the step
budget and match the independently calculated result.

## Approach

The graph receives each batch, inspects its current moisture, and conditionally
routes it to either `dry_batch` or `grade_batch`. The fix makes
`inspect_batch` return both the incremented inspection count and the current
approval decision:

```python
ready = MOISTURE_MIN <= state["moisture"] <= MOISTURE_MAX
return {"inspections": state["inspections"] + 1, "approved": ready}
```

Out-of-range batches lose 0.8 moisture per drying pass and are inspected
again. In-range batches go directly to grading.

## How it works

`route_after_inspection` reads `state["approved"]`. Persisting the decision in
the graph state means an in-range batch takes the grading branch instead of
repeating the drying branch forever. The existing graph structure is otherwise
preserved, and the evaluator verifies that only `inspect_batch` changed.

## Run it

From this directory, install the repository dependencies if needed and run:

```bash
python3 -m pip install langgraph
python3 evaluation_script.py
```

The evaluator loads `dataset/batches.json`, checks the protected graph
structure, invokes a fresh graph for every batch with an 18-step recursion
limit, and prints the flag only if every check passes.

## Evidence and limits

The evaluator independently recomputes the expected moisture, minimum drying
passes, and grade using fixed constants rather than trusting the submitted
graph's constants or approval flag:

| Batch | Starting moisture | Drying passes | Final moisture | Grade |
|---|---:|---:|---:|---:|
| nyamasheke | 12.0 | 0 | 12.0 | A |
| rutsiro | 11.0 | 0 | 11.0 | A |
| huye | 13.3 | 1 | 12.5 | A |
| nyabihu | 13.9 | 2 | 12.3 | A |
| muhanga | 14.9 | 3 | 12.5 | A |
| karongi | 15.0 | 4 | 11.8 | A |
| gicumbi | 16.5 | 5 | 12.5 | A |
| ngororero | 17.3 | 6 | 12.5 | A |

The worst case takes 15 node executions, leaving three executions of
headroom under the evaluator's 18-step budget. The evaluator also checks that
each batch was received, approved, and graded correctly. A local run requires
the `langgraph` package; without it, `evaluation_script.py` cannot start.

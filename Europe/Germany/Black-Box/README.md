# Black Box

## Task

Recover the structure of a hidden five-input function using at most 150 oracle queries. The submission must identify relevant variables, a pairwise interaction, and a discontinuity, then predict 300 challenge points. A score of at least 70 passes the organizer's hidden evaluation.

## Approach

We query a small factorial design around the centre, changing several inputs together. This lets us fit the linear effects and cross-products. We then query all 32 cube corners and subtract the smooth prediction. The remaining two output levels reveal the jump.

Next, we choose crossing edges that spread across the input space and bisect them to locate points on the switching surface. We fit the sparsest linear boundary consistent with those points and the corner observations. Finally, we refit the coefficients away from the boundary and return the structure, an expression, and predictions.

## Why it works

A centred product disappears when either interacting input stays at the middle. That is why changing one dial at a time can miss it. Factorial queries vary both inputs together. Subtracting the smooth part also separates the jump from ordinary changes in the output, so the boundary can be measured directly.

## Run it

Use Python 3.11 and install the required NumPy version:

```bash
pip install -r requirements.txt
python evaluation_script.py main.py --eval-type train --save results
python validate.py
```

Submit only **`main.py`**. It defines `solve(oracle, points)` and imports only NumPy and the standard library. The organizers run the hidden test. The archive does not contain `evaluator/hidden_function.py`, so that evaluation cannot be run locally.

## Evidence and limits

The official practice evaluation scores **100/100** using **136/150 queries**. The interaction and jump coefficients are recovered accurately, the normalized boundary offset error is about 0.0002, and the maximum prediction error is about 0.000001.

All **100 synthetic cases** also score 100/100, with at most 136 queries. They vary coefficients, interacting inputs, gate inputs, and positive or negative boundary directions while keeping the one-dial probing trap.

See the [query log](results/Black-Box/query_log.jsonl), [practice submission](results/Black-Box/submission.json), [synthetic results](results/synthetic_checks.csv), and [synthetic summary](results/synthetic_summary.json).

The solver assumes the documented family: linear effects, one pairwise product, and one jump across a linear boundary. Its smooth probe region must avoid the jump, or the alternate region must do so. Very small effects or extremely narrow switching regions can be difficult with rounded readings and a limited query budget. These local checks do not establish the hidden score or platform acceptance. The exposed flag constant in the supplied evaluator is not used by the solver.

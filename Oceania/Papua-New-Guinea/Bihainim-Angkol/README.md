# Bihainim Angkol (Follow Your Uncle)

## Task

Learn from the fishing logbook and improve on angkol's landed catch by at least 1.5 times on unseen days. The agent has 24 steps, a hold for six fish, and must return to the beach to score. The submitted file must answer every input state without importing the simulator.

## Approach

We first learn a simple imitation policy by taking the most common logged action in each observed state. Unseen states default to waiting. This gives a baseline, but it still spends too much time on the low-yield seagrass grounds.

We then use backward dynamic programming over all 6,048 states. For each action, we average the immediate landed fish and the best future catch across the possible bite and tide outcomes. We begin with zero future reward when time runs out, so fish left aboard have no value. Repeating this calculation for one through 24 remaining steps gives the complete policy.

The final policy is exported as a compressed table inside `main.py`. It reads `test-input.csv` and writes `test-output.csv` using only the standard library. It needs no model files, logbook or simulator at runtime.

## Why it works

The outer reef has a 90% bite chance, compared with 20% on the seagrass flats. Its extra travel is usually worth it. The policy accounts for the tide, limited hold and remaining return time instead of following one fixed route or fishing until the last step. The result maximises expected landed catch under the supplied transition model, rather than memorising practice seeds.

## Run it

No third-party packages are needed. From this task folder:

```bash
python train.py
python verify.py
```

Training rebuilds `main.py` and the evidence. Verification runs the exported file in a temporary directory containing only that file and the practice input, then checks the CSV and its catch.

For the supplied simulator's comparison:

```bash
cp train-input.csv test-input.csv
python main.py
python lagoon.py --policy test-output.csv --compare train-output.json
```

Upload only `main.py` to the platform.

## Evidence and limits

| Policy | Mean catch on 400 practice days |
| --- | ---: |
| Angkol's recorded catch | 2.4900 |
| Most-common-action imitation | 2.7650 |
| Dynamic programming | 5.9625 |

The final policy reaches **2.395 times** the recorded angkol average, above the required 1.5. On 2,000 additional seeds, 10001 through 12000, it averages **5.965 fish per day**. The model's exact expected catch from the starting state is **5.9575**.

`results/metrics.json` records these measurements. `results/fresh-days.csv` contains every additional seed's catch. `results/policy.csv` stores the improved policy, and `results/imitation.csv` stores the learned baseline. `results/standalone-check.json` confirms that the exported policy covers all states, preserves input order and passes the local comparison.

The submission passed the platform evaluation, as confirmed by our team. The exact hidden-day catch and ratio were not supplied. We do not have angkol's catches on the additional seeds, so their mean is a stability check rather than a measured hidden-day improvement ratio. The policy assumes the supplied map, bite probabilities and tide rules remain unchanged.

# The Douro Paradox

## Task

Find the column that reverses the apparent advantage of the new irrigation method: treatment must have a lower harvest success rate than control in every group. The flag is **`SIGMOID_DISTANCE_FROM_PORTO_KM`**.

## Approach

We compared the pooled success rates, checked each categorical column, and scanned numeric thresholds with at least 30 plots in each treatment arm on both sides. Only `distance_from_porto_km` produced a complete reversal within that search. We split it at 110 km, inside the observed gap between 104 and 114 km. Every cutoff inside that gap gives the same two groups.

## Why it works

The pooled rates use different group weights for treatment and control. About 92.38% of treatment plots are in the farther group, compared with 25.41% of control plots. Both methods succeed more often there. Treatment's concentration in that easier group makes its pooled rate higher even though it loses within both groups.

## Run it

The script uses only the Python standard library. From this folder, run:

```bash
python3 main.py
```

It reads `data/douro_vineyard_data.csv` and writes the counts, success rates, candidate checks, and flag into `results/`.

## Evidence and limits

| Group | Control successes / plots | Control rate | Treatment successes / plots | Treatment rate |
|---|---:|---:|---:|---:|
| Overall | 447 / 1,653 | 27.04% | 585 / 1,890 | 30.95% |
| Distance <=110 km | 312 / 1,233 | 25.30% | 36 / 144 | 25.00% |
| Distance >110 km | 135 / 420 | 32.14% | 549 / 1,746 | 31.44% |

Treatment leads overall by 3.91 percentage points but falls behind control in both distance groups. All 3,543 plots are included. Pooling reverses the comparison because the two treatment arms contain very different proportions of plots from the higher-success distance group.

The [overall rates](results/overall.csv) and [group rates](results/group_rates.csv) contain the exact counts. The [column checks](results/column_checks.csv) and [reversing thresholds](results/reversing_thresholds.csv) record the search. The search covers categorical levels and single numeric cutoffs, rather than every possible grouping. This descriptive reversal does not establish a causal effect or statistical significance. The threshold was identified from this dataset; platform acceptance has not been checked here.

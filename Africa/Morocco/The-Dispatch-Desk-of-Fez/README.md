# The Dispatch Desk of Fez

## Task

Route each request to a cheap guide costing 1 dirham or an expensive guide costing 25. On 500 hidden requests, at least 80% of answers must be correct and the total cost must stay below or equal to 10,000 dirham. Only the request text is available when choosing a guide.

## Approach

We compared both guides against the recorded reference answers and grouped requests by their structure. The router uses the cheap guide for direct price lookups, simple stock multiplications with both factors at most 12, and walking questions where an indirect path beats the direct connection. Other requests normally go to the expensive guide.

For walking questions, we build a small undirected graph and find the shortest path. This identifies the kind of reasoning needed; the script still returns only a guide choice.

Each category has a priority based on the extra accuracy the expensive guide provided in training. If too many requests need it, we send the highest priorities first. At most 78% of requests use the expensive guide, keeping the cost at or below 9,860 dirham for 500 requests. Equal priorities keep input order.

## Why it works

The larger guide helps most on tolls, closing times, comparisons and stock calculations with several steps. It adds little on simple lookups and small multiplications. Both guides struggle with indirect walking routes, so paying extra for those rarely helps. We spend the budget where the training answers show the biggest benefit, rather than treating every long question as difficult.

## Run it

Python 3.10 to 3.12 is supported. No packages or model downloads are needed. Run these commands from this task folder:

```bash
python evaluate.py
python -m unittest discover -p test_router.py
```

For a manual training check:

```bash
cp data/train-input.csv test-input.csv
python main.py
```

The script reads `test-input.csv` from the working directory and writes `routing_log.csv`, preserving request IDs and order. Evaluation uses a temporary directory and saves its training log under `results/`.

**Upload only `main.py` to the platform.** It contains all routing rules and does not read the training files at runtime.

## Evidence and limits

| Router | Training quality | Cost for 400 requests | Passes both gates |
| --- | ---: | ---: | --- |
| Always cheap | 33.75% | 400 | No |
| Always expensive | 85.50% | 10,000 | No |
| Our router | 84.50% | 7,672 | Yes |

`results/categories.csv` shows the guide accuracy by category. `results/train-routing.csv` records the training decisions, and `results/metrics.json` contains the overall metrics.

We also refitted category gains across 20 shuffled five-fold splits. Average quality was 83.525%, and 83 of 100 folds passed both gates. The weakest fold reached 73.75%. These are exploratory checks: we designed the categories and threshold after inspecting the full training set, so this is not independent validation. Details are in `results/split-checks.csv`.

Tests check the CSV contract, direct versus indirect routes, empty input and the budget on a batch of difficult requests. The submission passed the platform evaluation, as confirmed by our team. The platform did not provide the exact hidden-test quality or cost. A different mix of requests or different wording may reduce quality. Unknown templates get a medium priority. No hidden test or flag is supplied in the archive.

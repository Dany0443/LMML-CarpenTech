# Feather Stories

## Task

Find the six-letter ring of the White Stork missing from the register. The accepted answer is **`SIGMOID_ANSEMO`**.

## Approach

We compared the locations of ring-bearing stork sightings with the registered storks. One site in Cahul, `L5819951`, has 24 readings but no registered stork nearby.

Next, we used 2,998 sightings of registered birds to learn how letters are misread. The errors repeatedly stay within six groups: `AKRX`, `BGSZ`, `CDOQ`, `EFLP`, `HMNW`, and `IJTY`. The decoder fits six shared substitution patterns, converts each into a mapping between letters, and pools the remaining errors within each pattern. It scores the candidate ring under each pattern and returns the highest-scoring inscription: `ANSEMO`.

## Why it works

Repeated readings can contain the same systematic mistake, so voting for the most common letter can give the wrong answer. In the pattern supporting `ANSEMO`, the letters are commonly read like this:

| Real letter | Common reading |
|---|---|
| A | R |
| N | H |
| S | G |
| E | L |
| M | W |
| O | Q |

This explains why the raw readings favour `RHGLWQ`. Learning substitutions from registered birds lets us compare the underlying inscriptions instead of relying on raw letter counts.

## Run it

Use Python 3.11 and run these commands from this folder:

```bash
pip install -r requirements.txt
python main.py
```

`main.py` locates the missing bird and prepares the calibration readings. `decode.py` learns the reading patterns with 12 fixed initialisations. The scripts regenerate the evidence files and `results/flag.txt` directly from the two input CSVs, without reading a saved answer.

## Evidence and limits

The candidate has 24 checklists across 23 dates and 24 observer IDs. Its location is 74.34 km from the nearest registered stork; all other ring-bearing stork sightings are within 3.37 km of a registered location.

The [candidate observations](results/candidate_observations.csv), [site distances](results/site_distances.csv), [letter decoding](results/letter_decoding.csv), and [candidate scores](results/channel_candidates.csv) show the evidence. The [summary](results/summary.json) records the final answer.

The final model ranks `ANSEMO` first, and the team confirmed that Sigmoid accepted it. Its score is close to another candidate, so the statistical ranking alone is not conclusive. This model was selected during exploration and platform feedback, rather than evaluated on an untouched test set. The geographic association also assumes the register's home locations help identify the birds in calibration sightings.

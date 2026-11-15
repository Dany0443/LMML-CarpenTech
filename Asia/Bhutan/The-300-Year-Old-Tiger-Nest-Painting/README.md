# The 300-Year Old Tiger Nest Painting

## Task

Reassemble shuffled vertical image strips, then write `reconstructed_image.png` and `reconstructed_order.json`. The hidden evaluation requires at least 90% structural similarity, at least 90% correct ordering and a runtime below 20 seconds.

## Approach

We compare the rightmost pixel column of every strip with the leftmost column of every other strip. Their squared colour difference gives a seam cost. We sort all possible seams from cheapest to most expensive and join compatible path fragments.

A strip may have only one neighbour on each side. We also reject a join if it would form a cycle. After enough joins, one left-to-right chain contains every strip exactly once. We concatenate the original strips in that order and save their filenames as JSON. The code never reads the training answer or uses the numeric parts of filenames as positions.

## Why it works

Neighbouring columns of a continuous picture usually have similar colours and edges. The correct seams therefore tend to have lower costs than unrelated pairs. Comparing the full height makes individual noisy pixels less influential. Joining fragments globally also avoids relying on a guessed leftmost strip.

## Run it

Use Python 3.13 and the supplied dependencies:

```bash
python -m pip install -r requirements.txt
python verify.py
```

The verifier runs a copy of `main.py` with only an `input/` folder in a temporary directory, checks its outputs against the public reference and copies the reconstructed files into this task folder.

For a direct run, place the shuffled PNG strips in a folder named `input` in the working directory:

```bash
python main.py
```

Upload `main.py` for execution on the hidden strips. The two reconstruction files beside it show the public training result.

## Evidence and limits

The training reconstruction puts all 192 strips in the correct order and reproduces the reference image pixel for pixel. `results/training-check.json` records the measured order accuracy, SSIM, runtime and library versions. The generated PNG and JSON are saved alongside the script.

This is a greedy reconstruction method, not a proof of the globally minimum seam cost. Repeated patterns, flat regions, damaged edges or changes in strip orientation can make joins ambiguous. The supplied strips are clean vertical cuts with consistent orientation. The hidden evaluation has not yet been confirmed.

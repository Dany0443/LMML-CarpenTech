# Twins Separated at Birth

## Task

Find pairs of images that show the same scene, including images that have been mirrored or rotated.

## Approach

The script uses perceptual hashes to find likely matches, then checks each candidate with SIFT features and RANSAC. It uses the number and quality of matching features, image coverage, and correlation to decide whether to keep a pair. Accepted pairs are grouped, then expanded into the output CSV.

## Why it works

Perceptual hashes make the first search fast. SIFT and geometric checks reject many lookalikes that share only a small region. Rotated and mirrored hash variants help find the same scene in a different orientation.

## Run it

Install NumPy and OpenCV with SIFT support. From this folder, run:

```bash
python main.py /path/to/images-or-archive
python evaluate.py output.csv train-output.csv
```

The original image archive is not included in the repository, so a full rerun needs that archive.

## Evidence and limits

The saved practice output has precision 0.898, recall 0.831, and F1 0.863. These are not perfect results. The team reports platform acceptance. The saved pair files are `output.csv` and `train-output.csv`. A false match can join two unrelated groups; hard crops, major edits, and hash collisions can also cause errors.

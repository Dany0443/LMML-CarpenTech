# Looks like an anomaly!

## Task

Score one bunker sensor reading at a time. The detector must catch isolated spikes, gradual drifts, and changes that appear across several sensors, without seeing future readings.

## Approach

`Detector` keeps a separate running baseline for each zone. It compares each sensor with that baseline, then combines three signals: a large single-sensor deviation, a persistent smoothed deviation, and several moderate deviations occurring together. Suspicious readings have limited effect on future baselines.

## Why it works

Standardised deviations put sensors with different units on a comparable scale. The single-sensor signal catches spikes, smoothing catches sustained drift, and combining the largest sensor deviations catches collective events. The detector stores summary state and uses no future readings.

## Run it

From this folder, run the offline evaluation and tests:

```bash
python3.11 evaluate.py
python3.11 -m unittest discover -p 'test_*.py'
```

Submit `detector.py` to the platform. The offline evaluator reads labels; `Detector.update()` receives one reading per call.

## Evidence and limits

On the supplied 3,600 readings, the local score is 98.63/100, with 284 true positives, six false positives, and three false negatives. Point and collective recall are 100%; drift recall is 98.33%. Ten tests cover synthetic anomalies, ordinary cycles, new zones, invalid values, and bounded state. The team reports platform acceptance. Both halves of the stream were inspected during development, so the later-half score is not an untouched test. Anomalies during warm-up, weak changes, and genuine permanent shifts in normal conditions can still be missed or misclassified.

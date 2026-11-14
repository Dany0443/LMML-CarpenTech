# Looks like an anomaly!

We needed to detect abnormal sensor readings as they arrive, without looking at future measurements. We learned a separate baseline for each bunker zone using its first 40 readings, then checked for sudden spikes, slow drifts, and several sensors changing together.

We also learned each zone's normal range so ordinary temperature cycles would not trigger false alarms. Suspicious readings have limited influence on the baseline, which helps prevent the detector from accepting an ongoing problem as normal.

On the supplied 3,600 readings, we obtained **98.63/100**, with six false alarms and three missed anomalies. Ten tests pass under Python 3.11. These are local development results; the hidden evaluation still needs platform validation, and anomalies during the initial warm-up can be missed.


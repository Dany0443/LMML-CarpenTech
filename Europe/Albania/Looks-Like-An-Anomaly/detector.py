import math

SENSORS = (
    'temperature_c', 'humidity_pct', 'vibration_g',
    'acoustic_db', 'air_pressure_kpa', 'co2_ppm',
)


class Detector:
    def __init__(self):
        self.zones = {}

    def update(self, reading: dict) -> float:
        try:
            values = [float(reading[sensor]) for sensor in SENSORS]
        except (KeyError, TypeError, ValueError, OverflowError):
            return 1.0
        if not all(math.isfinite(value) for value in values):
            return 1.0

        zone = reading.get('bunker_zone')
        if zone not in self.zones:
            self.zones[zone] = {
                'count': 0,
                'means': values[:],
                'variances': [0.] * 6,
                'smooth': [0.] * 6,
                'previous': values[:],
                'drifting': [False] * 6,
                'centers': values[:],
                'spread': [0.] * 6,
                'accepted': [0] * 6,
            }
        baseline = self.zones[zone]
        baseline['count'] += 1
        count = baseline['count']
        means = baseline['means']
        variances = baseline['variances']
        smooth = baseline['smooth']
        previous = baseline['previous']
        drifting = baseline['drifting']
        centers = baseline['centers']
        spread = baseline['spread']
        accepted = baseline['accepted']

        if count <= 40:
            for channel, value in enumerate(values):
                residual = value - means[channel]
                means[channel] += residual / count
                variances[channel] += residual * (value - means[channel])
            if count == 40:
                centers[:] = means
                spread[:] = variances
                accepted[:] = [40] * 6
                variances[:] = [variance / 39. for variance in variances]
            previous[:] = values
            return 0.0

        residuals = [value - center for value, center in zip(values, means)]
        scales = [max(math.sqrt(max(variance, spread[channel] / (accepted[channel] - 1), 0.)), 1e-9)
                  for channel, variance in enumerate(variances)]
        zscores = [residual / scale for residual, scale in zip(residuals, scales)]
        smooth[:] = [.8 * average + .2 * max(-6., min(6., z))
                     for average, z in zip(smooth, zscores)]
        drift = max(abs(average) if average * z > 0. and abs(z) > 1. else 0.
                    for average, z in zip(smooth, zscores))
        for channel, (average, z) in enumerate(zip(smooth, zscores)):
            if average * z <= 0. or abs(z) <= 1. or abs(average) < 1.5:
                drifting[channel] = False
            elif abs(average) >= 2.:
                drifting[channel] = True
        largest = sorted(zscores, key=abs)[-3:]
        joint = math.sqrt(sum(min(abs(z), 4.) ** 2 for z in largest))
        strength = max(max(map(abs, zscores)) / 5.5, drift / 2.0,
                       joint / 5.5, float(any(drifting)))
        flagged = strength >= 1.

        for channel, residual in enumerate(residuals):
            z = abs(zscores[channel])
            change = values[channel] - previous[channel]
            if change * change < 16. * variances[channel]:
                variances[channel] = .98 * variances[channel] + .01 * change * change
            if z < 1.5:
                accepted[channel] += 1
                displacement = values[channel] - centers[channel]
                centers[channel] += displacement / accepted[channel]
                spread[channel] += displacement * (values[channel] - centers[channel])
            if z < 3. and not (flagged and z > 1.5):
                means[channel] += .02 * residual
        previous[:] = values
        return min(1.0, .5 * strength)

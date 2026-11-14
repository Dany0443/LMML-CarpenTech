import math
import pickle
import random
import unittest

from detector import Detector

SENSORS = ('temperature_c', 'humidity_pct', 'vibration_g', 'acoustic_db', 'air_pressure_kpa', 'co2_ppm')
CENTERS = (13., 72., .12, 32., 101.5, 510.)
NOISE = (.2, .6, .01, .35, .05, 10.)


def reading(rng, offsets=None, zone='unseen_zone'):
    shifts = offsets or [0.] * 6
    sample = {'timestamp': '2030-01-01', 'bunker_zone': zone}
    for sensor, center, noise, shift in zip(SENSORS, CENTERS, NOISE, shifts):
        sample[sensor] = center + noise * (rng.gauss(0., 1.) + shift)
    return sample


def warmed(seed=104):
    rng = random.Random(seed)
    detector = Detector()
    for _ in range(200):
        detector.update(reading(rng))
    return detector, rng


class DetectorTests(unittest.TestCase):
    def test_point_spike_in_every_sensor(self):
        for channel in range(6):
            with self.subTest(sensor=SENSORS[channel]):
                detector, rng = warmed()
                shifts = [0.] * 6
                shifts[channel] = 20.
                self.assertGreaterEqual(detector.update(reading(rng, shifts)), .5)

    def test_gradual_drift_is_detected(self):
        detector, rng = warmed()
        flags = []
        for step in range(60):
            shifts = [0., 8. * (step + 1) / 60., 0., 0., 0., 0.]
            flags.append(detector.update(reading(rng, shifts)) >= .5)
        self.assertGreaterEqual(sum(flags), 40)

    def test_joint_moderate_changes_are_detected(self):
        for shift in (2., 3.):
            with self.subTest(shift=shift):
                detector, rng = warmed()
                shifts = [0., 0., shift, shift, 0., shift]
                flags = [detector.update(reading(rng, shifts)) >= .5 for _ in range(40)]
                self.assertGreaterEqual(sum(flags), 28)

    def test_stationary_normal_stream_has_few_false_alarms(self):
        rng = random.Random(804)
        detector = Detector()
        scores = [detector.update(reading(rng)) for _ in range(3000)]
        self.assertTrue(all(math.isfinite(score) and 0. <= score <= 1. for score in scores))
        self.assertLess(sum(score >= .5 for score in scores[40:]), 30)

    def test_zone_names_and_timestamps_do_not_encode_predictions(self):
        rng = random.Random(12)
        first, second = Detector(), Detector()
        for _ in range(300):
            sample = reading(rng)
            renamed = dict(sample, bunker_zone='another_unknown_zone', timestamp='2045-12-31')
            self.assertEqual(first.update(sample), second.update(renamed))

    def test_different_zones_keep_separate_baselines(self):
        detector = Detector()
        rng = random.Random(28)
        flags = 0
        for _ in range(300):
            for zone, shifts in [('room_a', [0.] * 6), ('room_b', [25.] * 6)]:
                flags += detector.update(reading(rng, shifts, zone)) >= .5
        self.assertLess(flags, 10)

    def test_invalid_values_do_not_corrupt_the_baseline(self):
        detector, rng = warmed()
        before = pickle.dumps(detector)
        for invalid in (float('nan'), float('inf'), None, 'broken'):
            sample = reading(rng)
            sample['co2_ppm'] = invalid
            self.assertEqual(detector.update(sample), 1.)
            self.assertEqual(pickle.dumps(detector), before)

    def test_history_does_not_grow_per_reading(self):
        detector, rng = warmed()
        before = len(pickle.dumps(detector))
        for _ in range(5000):
            detector.update(reading(rng))
        self.assertLess(len(pickle.dumps(detector)), before + 100)

    def test_normal_cycles_survive_a_long_zone_gap(self):
        detector = Detector()
        rng = random.Random(913)
        flags = 0
        for tick in range(1800):
            zone = 'first_room' if tick < 600 or tick >= 1200 else 'second_room'
            shifts = [2. * math.sin(2. * math.pi * tick / 800.), 0., 0., 0.,
                      2. * math.sin(2. * math.pi * tick / 1100.), 0.]
            flags += detector.update(reading(rng, shifts, zone)) >= .5
        self.assertLess(flags, 36)

    def test_sensor_scaling_and_offsets_do_not_encode_predictions(self):
        rng = random.Random(388)
        first, second = Detector(), Detector()
        for tick in range(400):
            shifts = [0., max(0., tick - 300) * .1, 0., 0., 0., 0.]
            sample = reading(rng, shifts)
            transformed = dict(sample)
            for channel, sensor in enumerate(SENSORS):
                transformed[sensor] = sample[sensor] * (channel + 2.) + 100.
            self.assertAlmostEqual(first.update(sample), second.update(transformed), places=7)

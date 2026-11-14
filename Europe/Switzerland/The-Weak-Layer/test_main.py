import itertools
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd


FOLDER = Path(__file__).resolve().parent


class DecoderTests(unittest.TestCase):
    def decode(self, pits):
        with tempfile.TemporaryDirectory() as directory:
            working = Path(directory)
            pits.to_csv(working / 'test-input.csv', index=False)
            execution = subprocess.run(
                [sys.executable, str(FOLDER / 'main.py')],
                cwd=working, capture_output=True, text=True, check=True,
            )
            self.assertEqual(execution.stdout, '')
            return (working / 'test-output.txt').read_text()

    def snow_pits(self, word, inverted=False):
        profiles = []
        for bits in itertools.product([0, 1], repeat=len(word)):
            copies = 1
            for position, bit in enumerate(bits):
                if bit:
                    copies *= position + 1
            profiles.extend([bits] * copies)
        readings = np.array(profiles)
        columns = [f'{letter.lower()}_measurement_{position}' for position, letter in enumerate(word)]
        pits = pd.DataFrame(readings, columns=columns)
        pits['slab_released'] = readings.all(axis=1).astype(int)
        if inverted:
            pits[columns] = 1 - pits[columns]
        shuffled = np.random.default_rng(17).permutation(pits.columns)
        return pits[shuffled].sample(frac=1, random_state=29).reset_index(drop=True)

    def test_practice_matches_the_provided_answer(self):
        pits = pd.read_csv(FOLDER / 'data/train-input.csv')
        expected = (FOLDER / 'data/train-output.txt').read_text().strip()
        self.assertEqual(self.decode(pits), expected + '\n')

    def test_unseen_word_with_shuffled_columns_and_target_position(self):
        self.assertEqual(self.decode(self.snow_pits('FROST')), 'SIGMOID_FROST\n')

    def test_repeated_letters_are_preserved(self):
        self.assertEqual(self.decode(self.snow_pits('SLEET')), 'SIGMOID_SLEET\n')

    def test_informative_branch_can_be_on_the_left(self):
        self.assertEqual(self.decode(self.snow_pits('SNOW', inverted=True)), 'SIGMOID_SNOW\n')

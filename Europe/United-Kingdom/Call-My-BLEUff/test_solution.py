import unittest

from solve import decode, defect_score
from slots import expected, slots


class SolutionTests(unittest.TestCase):
    def test_base_three_decode_and_invalid_letter(self):
        self.assertEqual(decode([1, 1, 1, 3, 3, 2]), 'AZ')
        with self.assertRaises(ValueError):
            decode([3, 3, 3])
        with self.assertRaises(ValueError):
            decode([1, 2])

    def test_omission_and_padding_are_worse_than_complete_description(self):
        good = 'Aromi is a cheap Japanese coffee shop by the riverside.'
        refs = [[good] * 5] * 40
        intact = defect_score([good] * 40, refs)
        omitted = defect_score(['Aromi is a coffee shop.'] + [good] * 39, refs)
        padded = defect_score([good + ' ' + good] * 40, refs)
        self.assertLess(intact, omitted)
        self.assertLess(intact, padded)

    def test_one_noisy_reference_does_not_set_consensus(self):
        refs = ['Aromi is a cheap coffee shop.'] * 4 + ['Aromi is an expensive pub.']
        concepts = expected(refs)
        self.assertIn('price:low', concepts)
        self.assertNotIn('price:high', concepts)

    def test_nearby_venue_does_not_invent_food_and_negation_is_preserved(self):
        concepts = slots('Cotto is not family-friendly near Raja Indian Cuisine.')
        self.assertIn('family:no', concepts)
        self.assertNotIn('family:yes', concepts)
        self.assertNotIn('food:indian', concepts)

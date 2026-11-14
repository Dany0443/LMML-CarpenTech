import unittest
from pathlib import Path

import pandas as pd

from solution import build_board


class BoardTests(unittest.TestCase):
    def setUp(self):
        self.guide = pd.DataFrame({
            'List_no': [6, 5, 4, 3, 2, 1],
            'Product': ['Barquillos', 'Saffron', 'Paella', 'Turron', 'Churros', 'Jamon'],
            'Fair_min_eur': ['2.10'] * 6,
            'Fair_max_eur': ['3.30'] * 6,
        })

    def receipts(self, totals, quantities, pilots):
        return pd.DataFrame({
            'Receipt_id': [f'R{index}' for index in range(len(totals))],
            'Pilot_no': pilots,
            'Product': ['Jamon'] * len(totals),
            'Quantity': quantities,
            'Total_paid_eur': totals,
        })

    def test_inclusive_unit_price_boundaries_with_fractional_quantities(self):
        purchases = self.receipts(['0.21', '0.33', '0.209', '0.331'], ['0.1'] * 4, [1, 2, 3, 4])
        board, audit = build_board(purchases, self.guide)
        self.assertEqual(audit.Fair.tolist(), [True, True, False, False])
        self.assertEqual(board.loc[1, [1, 2, 3, 4]].tolist(), [1, 1, 0, 0])

    def test_repeated_receipts_use_any_fair_purchase(self):
        purchases = self.receipts(['5.00', '6.00', '1.00'], ['2', '2', '2'], [7, 7, 7])
        board, audit = build_board(purchases, self.guide)
        self.assertEqual(audit.Fair.sum(), 2)
        self.assertEqual(board.loc[1, 7], 1)
        self.assertEqual(board.to_numpy().sum(), 1)

    def test_all_pilots_and_office_rows_remain_in_numeric_order(self):
        purchases = self.receipts(['5.00', '5.00'], ['2', '2'], [21, 1])
        board, audit = build_board(purchases, self.guide)
        self.assertEqual(board.shape, (6, 21))
        self.assertEqual(board.index.tolist(), list(range(1, 7)))
        self.assertEqual(board.columns.tolist(), list(range(1, 22)))
        self.assertEqual(audit.Receipt_id.tolist(), purchases.Receipt_id.tolist())
        self.assertEqual(board.loc[6].sum(), 0)

    def test_invalid_quantities_pilots_and_products_are_rejected(self):
        purchases = self.receipts(['5.00'], ['2'], [1])
        for broken in [purchases.assign(Quantity='0'), purchases.assign(Quantity='-1'),
                       purchases.assign(Pilot_no=22), purchases.assign(Product='Unknown')]:
            with self.subTest(receipt=broken.to_dict()):
                with self.assertRaises(ValueError):
                    build_board(broken, self.guide)

    def test_supplied_receipts_reproduce_the_board_totals(self):
        folder = Path(__file__).resolve().parent / 'data'
        purchases = pd.read_csv(folder / 'purchases.csv', dtype=str)
        guide = pd.read_csv(folder / 'office_price_guide.csv', dtype=str)
        board, audit = build_board(purchases, guide)
        self.assertEqual(len(audit), 577)
        self.assertEqual(audit.Fair.sum(), 77)
        self.assertEqual(board.to_numpy().sum(), 57)
        self.assertEqual(audit.groupby('List_no').Fair.sum().tolist(), [12, 8, 13, 13, 10, 21])

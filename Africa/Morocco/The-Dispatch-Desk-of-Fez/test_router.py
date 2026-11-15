import csv
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

SCRIPT = Path(__file__).with_name('main.py')


class RouterTests(unittest.TestCase):
    def route(self, prompts):
        with tempfile.TemporaryDirectory() as scratch:
            folder = Path(scratch)
            with (folder / 'test-input.csv').open('w', newline='') as stream:
                writer = csv.writer(stream)
                writer.writerow(['request_id', 'prompt'])
                writer.writerows((str(index), prompt) for index, prompt in enumerate(prompts))
            subprocess.run([sys.executable, str(SCRIPT)], cwd=folder, check=True)
            with (folder / 'routing_log.csv').open(newline='') as stream:
                return list(csv.DictReader(stream))

    def test_budget_on_all_difficult_requests(self):
        routes = self.route(['A new difficult question'] * 500)
        self.assertEqual([row['request_id'] for row in routes], list(map(str, range(500))))
        self.assertTrue(all(row['model'] in {'cheap', 'expensive'} for row in routes))
        self.assertLessEqual(sum(25 if row['model'] == 'expensive' else 1 for row in routes), 10000)

    def test_direct_and_indirect_walks(self):
        prompts = [
            'Walking times in the medina: A-B 3 minutes, B-C 4 minutes, A-C 9 minutes. What is the shortest walk from A to C, in minutes?',
            'Walking times in the medina: A-B 3 minutes, B-C 4 minutes, A-C 6 minutes. What is the shortest walk from A to C, in minutes?',
            'The Henna board lists mint at 20 dirham. What does mint cost?',
            'The Bab Rcif gate charges 9 dirham for every cart. A caravan brings 10 carts through. What is the total toll?',
        ]
        self.assertEqual([row['model'] for row in self.route(prompts)], ['cheap', 'expensive', 'cheap', 'expensive'])

    def test_empty_batch(self):
        self.assertEqual(self.route([]), [])

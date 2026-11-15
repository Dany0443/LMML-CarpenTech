import ast
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path

import lagoon

ROOT = Path(__file__).resolve().parent
votes = defaultdict(Counter)
with (ROOT / 'dataset/logbook.csv').open(newline='') as stream:
    for row in csv.DictReader(stream):
        state = tuple(int(row[key]) for key in ['row', 'col', 'hold', 'steps_left'])
        votes[state][row['action']] += 1
clone = {state: choices.most_common(1)[0][0] for state, choices in votes.items()}
imitation = {state: clone.get(state, 'WAIT') for state in lagoon.all_states()}
cells = [(row, col) for row in range(lagoon.ROWS) for col in range(lagoon.COLS) if lagoon.is_water(row, col)]
values = {(row, col, hold): 0.0 for row, col in cells for hold in range(lagoon.HOLD_CAPACITY + 1)}
policy = {}
for remaining in range(1, lagoon.STEPS_PER_DAY + 1):
    next_values = {}
    for row, col in cells:
        bite_probability = lagoon.BITE_CHANCE.get(lagoon.GRID[row][col], 0.0)
        for hold in range(lagoon.HOLD_CAPACITY + 1):
            best = -1.0
            choice = None
            for action in lagoon.ACTIONS:
                expected = 0.0
                for bite, bite_weight in [(False, 1 - bite_probability), (True, bite_probability)]:
                    for tide, tide_weight in [(False, 1 - lagoon.TIDE_CHANCE), (True, lagoon.TIDE_CHANCE)]:
                        destination_row, destination_col, carried, landed = lagoon._advance(row, col, hold, action, bite, tide)
                        expected += bite_weight * tide_weight * (landed + values[destination_row, destination_col, carried])
                if expected > best + 1e-10:
                    best, choice = expected, action
            next_values[row, col, hold] = best
            policy[row, col, hold, remaining] = choice
    values = next_values
lagoon.export_main(policy, str(ROOT / 'main.py'))
script = ROOT / 'main.py'
syntax = ast.parse(script.read_text())
if isinstance(syntax.body[0], ast.Expr) and isinstance(syntax.body[0].value, ast.Constant):
    syntax.body.pop(0)
script.write_text(ast.unparse(syntax) + '\n')
lagoon.write_policy(policy, str(ROOT / 'results/policy.csv'))
lagoon.write_policy(imitation, str(ROOT / 'results/imitation.csv'))
reference = json.loads((ROOT / 'train-output.json').read_text())
practice_mean = sum(lagoon.play_day(policy, seed)[0] for seed in reference['days']) / len(reference['days'])
clone_mean = sum(lagoon.play_day(imitation, seed)[0] for seed in reference['days']) / len(reference['days'])
fresh_days = range(10001, 12001)
fresh_catch = [lagoon.play_day(policy, seed)[0] for seed in fresh_days]
clone_fresh = sum(lagoon.play_day(imitation, seed)[0] for seed in fresh_days) / len(fresh_days)
with (ROOT / 'results/fresh-days.csv').open('w', newline='') as stream:
    writer = csv.writer(stream)
    writer.writerow(['seed', 'landed'])
    writer.writerows(zip(fresh_days, fresh_catch))
metrics = {'practice_days': len(reference['days']), 'angkol_mean': reference['angkol_mean'],
           'imitation_practice_mean': clone_mean, 'policy_practice_mean': practice_mean,
           'practice_ratio': practice_mean / reference['angkol_mean'],
           'required_ratio': reference['target_ratio'], 'fresh_days': len(fresh_days),
           'fresh_seed_range': [10001, 12000], 'policy_fresh_mean': sum(fresh_catch) / len(fresh_catch),
           'imitation_fresh_mean': clone_fresh, 'optimal_expected_catch': values[*lagoon.START, 0],
           'states': len(policy), 'observed_imitation_states': len(clone)}
(ROOT / 'results/metrics.json').write_text(json.dumps(metrics, indent=2) + '\n')
print(json.dumps(metrics, indent=2))

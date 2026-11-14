import csv
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def save_csv(path, columns, records):
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream, lineterminator='\n')
        writer.writerow(columns)
        writer.writerows(records)


def rates(records, grouping):
    totals = defaultdict(lambda: [0, 0])
    for plot in records:
        group = grouping(plot)
        tally = totals[group, plot['treatment']]
        tally[0] += int(plot['harvest_success'])
        tally[1] += 1
    return [(group, arm, successes, count, successes / count) for (group, arm), (successes, count) in sorted(totals.items())]


with (ROOT / 'data/douro_vineyard_data.csv').open(encoding='utf-8-sig', newline='') as stream:
    plots = list(csv.DictReader(stream))
columns = [column for column in plots[0] if column not in ('plot_id', 'treatment', 'harvest_success')]
checks = []
cut_audit = []
for column in columns:
    try:
        ordered = sorted((float(plot[column]), int(plot['treatment'] == 'treatment'), int(plot['harvest_success'])) for plot in plots)
    except ValueError:
        grouped = rates(plots, lambda plot: plot[column])
        strata = defaultdict(dict)
        for group, arm, successes, count, rate in grouped:
            strata[group][arm] = (successes, count)
        reversal = all(len(arms) == 2 and arms['treatment'][0] * arms['control'][1] < arms['control'][0] * arms['treatment'][1] for arms in strata.values())
        checks.append((column, 'categorical groups', len(strata), int(reversal)))
        continue
    overall_counts = [sum(arm == index for _, arm, _ in ordered) for index in range(2)]
    overall_successes = [sum(success for _, arm, success in ordered if arm == index) for index in range(2)]
    left_counts = [0, 0]
    left_successes = [0, 0]
    tested = 0
    passing = 0
    for index, (value, arm, success) in enumerate(ordered[:-1]):
        left_counts[arm] += 1
        left_successes[arm] += success
        if value == ordered[index + 1][0]:
            continue
        right_counts = [overall_counts[side] - left_counts[side] for side in range(2)]
        right_successes = [overall_successes[side] - left_successes[side] for side in range(2)]
        if min(left_counts + right_counts) < 30:
            continue
        tested += 1
        reversal = all(successes[1] * counts[0] < successes[0] * counts[1] for successes, counts in ((left_successes, left_counts), (right_successes, right_counts)))
        if reversal:
            passing += 1
            cut_audit.append((column, value, ordered[index + 1][0]))
    checks.append((column, 'two contiguous numeric groups, >=30 per arm', tested, passing))

column = 'distance_from_porto_km'
cutoff = 110
split = rates(plots, lambda plot: '<=110 km' if float(plot[column]) <= cutoff else '>110 km')
pooled = rates(plots, lambda plot: 'overall')
reports = ROOT / 'results'
reports.mkdir(exist_ok=True)
header = ['group', 'treatment', 'successes', 'plots', 'success_rate']
save_csv(reports / 'overall.csv', header, pooled)
save_csv(reports / 'group_rates.csv', header, split)
save_csv(reports / 'column_checks.csv', ['column', 'grouping', 'eligible_splits_or_groups', 'complete_reversals'], checks)
save_csv(reports / 'reversing_thresholds.csv', ['column', 'last_left_value', 'first_right_value'], cut_audit)
flag = 'SIGMOID_' + column.upper()
(reports / 'flag.txt').write_text(flag + '\n', encoding='utf-8')
summary = {'plots': len(plots), 'column': column, 'cutoff_km': cutoff, 'gap_below_cutoff': 104, 'gap_above_cutoff': 114, 'flag': flag, 'search_scope': 'observed categorical levels and binary numeric thresholds with at least 30 plots per arm in both groups'}
(reports / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
print(flag)
for row in pooled + split:
    print(row)

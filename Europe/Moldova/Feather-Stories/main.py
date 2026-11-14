import csv
import json
from collections import Counter, defaultdict
from math import asin, cos, radians, sin, sqrt
from pathlib import Path

from decode import decode_ring

ROOT = Path(__file__).resolve().parent
SPECIES = 'Ciconia ciconia'


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def save_csv(path, columns, records):
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream, lineterminator='\n')
        writer.writerow(columns)
        writer.writerows(records)


def distance(latitude, longitude, bird):
    latitude, longitude = radians(float(latitude)), radians(float(longitude))
    home_latitude = radians(float(bird['ringed_latitude']))
    home_longitude = radians(float(bird['ringed_longitude']))
    arc = sin((latitude - home_latitude) / 2) ** 2
    arc += cos(latitude) * cos(home_latitude) * sin((longitude - home_longitude) / 2) ** 2
    return 6371.0088 * 2 * asin(sqrt(min(1, arc)))


def consensus(readings):
    votes = [Counter(reading[position] for reading in readings if reading[position] != '?') for position in range(6)]
    if any(not vote or (len(vote) > 1 and vote.most_common(2)[0][1] == vote.most_common(2)[1][1]) for vote in votes):
        raise ValueError('Consensus has an empty or tied position')
    return ''.join(vote.most_common(1)[0][0] for vote in votes), votes


observations = read_csv(ROOT / 'data/observations.csv')
register = read_csv(ROOT / 'data/ring_register.csv')
registered = [bird for bird in register if bird['species'] == SPECIES]
storks = [row for row in observations if row['species'] == SPECIES and row['ring_read']]
sites = defaultdict(list)
audit = []
for row in storks:
    nearest = min(registered, key=lambda bird: distance(row['latitude'], row['longitude'], bird))
    kilometres = distance(row['latitude'], row['longitude'], nearest)
    sites[row['locality_id']].append((row, kilometres))
    audit.append((row['record_id'], row['locality_id'], row['ring_read'], nearest['ring'], kilometres))
site_distances = sorted((min(km for _, km in records), site) for site, records in sites.items())
gaps = [(site_distances[index + 1][0] - site_distances[index][0], index) for index in range(len(site_distances) - 1)]
_, gap_index = max(gaps)
threshold = (site_distances[gap_index][0] + site_distances[gap_index + 1][0]) / 2
candidate_sites = [site for km, site in site_distances if km > threshold]
if len(candidate_sites) != 1:
    raise ValueError('Geographic separation does not isolate one candidate site')
site = candidate_sites[0]
candidate = sorted((row for row, _ in sites[site]), key=lambda row: (row['date'], row['time'], row['record_id']))
modal_ring, votes = consensus([row['ring_read'] for row in candidate])
calibration = []
by_species = defaultdict(list)
for bird in register:
    by_species[bird['species']].append(bird)
for row in observations:
    if not row['ring_read'] or row['species'] not in by_species or row['locality_id'] == site:
        continue
    nearest = min(by_species[row['species']], key=lambda bird: distance(row['latitude'], row['longitude'], bird))
    if distance(row['latitude'], row['longitude'], nearest) < 3.5:
        calibration.append((nearest['ring'], row['ring_read']))
ring, alternatives, starts, transformations = decode_ring(calibration, [row['ring_read'] for row in candidate])
assert ring not in {bird['ring'] for bird in register}
observers = sorted({observer for row in candidate for observer in row['observers'].split('|')})
reports = ROOT / 'results'
reports.mkdir(exist_ok=True)
save_csv(reports / 'candidate_observations.csv', list(candidate[0]), [list(row.values()) for row in candidate])
save_csv(reports / 'stork_register_distances.csv', ['record_id', 'locality_id', 'ring_read', 'nearest_registered_ring', 'distance_km'], audit)
save_csv(reports / 'site_distances.csv', ['locality_id', 'ring_readings', 'minimum_registered_distance_km'], [(name, len(sites[name]), km) for km, name in site_distances])
save_csv(reports / 'channel_candidates.csv', ['channel', 'candidate', 'log_likelihood'], alternatives)
save_csv(reports / 'decoder_starts.csv', ['seed', 'calibration_log_likelihood'], starts)
save_csv(reports / 'letter_decoding.csv', ['position', 'inferred_true_letter', 'modal_read_letter', 'channel_probability'], transformations)
save_csv(reports / 'letter_votes.csv', ['position', 'letter', 'votes'], [(position, letter, count) for position, vote in enumerate(votes, 1) for letter, count in vote.most_common()])
summary = {
    'ring': ring,
    'flag': 'SIGMOID_' + ring,
    'raw_consensus': modal_ring,
    'platform_status': 'accepted, reported by user',
    'calibration_readings': len(calibration),
    'candidate_locality': site,
    'district': candidate[0]['district'],
    'candidate_readings': len(candidate),
    'candidate_dates': len({row['date'] for row in candidate}),
    'candidate_checklists': len({row['checklist_id'] for row in candidate}),
    'distinct_observers': len(observers),
    'nearest_registered_stork_km': min(km for _, km in sites[site]),
    'maximum_other_stork_distance_km': max(
        km for name, records in sites.items() if name != site for _, km in records
    ),
    'exact_raw_consensus_readings': sum(row['ring_read'] == modal_ring for row in candidate),
}
(reports / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
(reports / 'flag.txt').write_text(summary['flag'] + '\n', encoding='utf-8')
print(json.dumps(summary, indent=2))

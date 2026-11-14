import csv
import json
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def normalise(name):
    name = unicodedata.normalize('NFKC', name).strip().rstrip('.').strip().casefold()
    return ''.join(letter for letter in unicodedata.normalize('NFKD', name) if not unicodedata.combining(letter))


def save_csv(path, columns, records):
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        writer.writerows(records)


with (ROOT / 'data/routes.csv').open(encoding='utf-8-sig', newline='') as stream:
    routes = list(csv.DictReader(stream))
counts = Counter(route[column] for route in routes for column in ('from_town', 'to_town'))
spellings = defaultdict(list)
for name in counts:
    spellings[normalise(name)].append(name)
canonical = {}
for key, variants in spellings.items():
    preferred = max(variants, key=lambda name: (sum(unicodedata.combining(letter) != 0 for letter in unicodedata.normalize('NFD', name)), counts[name]))
    canonical[key] = unicodedata.normalize('NFC', preferred.strip().rstrip('.').strip().title())
with (ROOT / 'town_aliases.csv').open(encoding='utf-8', newline='') as stream:
    for pair in csv.DictReader(stream):
        canonical[normalise(pair['finnish'])] = pair['finnish']
        canonical[normalise(pair['swedish'])] = pair['finnish']

connections = set()
towns = set()
audit = []
self_routes = 0
for route in routes:
    origin = canonical[normalise(route['from_town'])]
    destination = canonical[normalise(route['to_town'])]
    towns.update((origin, destination))
    edge = tuple(sorted((origin, destination)))
    if origin == destination:
        status = 'self_route'
        self_routes += 1
    elif edge in connections:
        status = 'duplicate'
    else:
        status = 'kept'
        connections.add(edge)
    audit.append((route['log_id'], origin, destination, status))

degrees = Counter(town for edge in connections for town in edge)
ranking = sorted(((town, degrees[town]) for town in towns), key=lambda entry: (-entry[1], entry[0]))
assert len(towns) == 116, f'Unexpected town count: {len(towns)}'
assert len(connections) == 448, f'Unexpected connection count: {len(connections)}'
assert len({degree for _, degree in ranking[:8]}) == 8
assert sum(degrees.values()) == 2 * len(connections)
flag = 'SIGMOID_' + ''.join(town[0] for town, _ in ranking[:7]).upper()
reports = ROOT / 'results'
reports.mkdir(exist_ok=True)
save_csv(reports / 'town_mapping.csv', ['raw_name', 'normalised_key', 'canonical_town', 'occurrences'], [(name, normalise(name), canonical[normalise(name)], counts[name]) for name in sorted(counts)])
save_csv(reports / 'clean_connections.csv', ['town_a', 'town_b'], sorted(connections))
save_csv(reports / 'ranking.csv', ['rank', 'town', 'distinct_connections'], [(rank, town, degree) for rank, (town, degree) in enumerate(ranking, 1)])
save_csv(reports / 'route_audit.csv', ['log_id', 'from_town', 'to_town', 'status'], audit)
summary = {'raw_rows': len(routes), 'raw_name_variants': len(counts), 'clean_towns': len(towns), 'clean_connections': len(connections), 'duplicate_rows': sum(row[3] == 'duplicate' for row in audit), 'self_routes': self_routes, 'flag': flag}
(reports / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
(reports / 'flag.txt').write_text(flag + '\n', encoding='utf-8')
print(json.dumps(summary, indent=2))
for rank, (town, degree) in enumerate(ranking[:8], 1):
    print(f'{rank}. {town}: {degree}')

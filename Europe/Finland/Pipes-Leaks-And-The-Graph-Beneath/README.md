# Pipes, Leaks, and the Graph Beneath

## Task

Clean the route log, rebuild the town network, and rank towns by their number of distinct connections. The first letters of the seven busiest towns form **`SIGMOID_HOTLINE`**.

## Approach

We normalise Unicode, whitespace, case, trailing full stops, and missing accents. We then merge Finnish and Swedish names for the same town using `town_aliases.csv`. The bilingual pairs come from the [Institute for the Languages of Finland's municipality name list](https://kielitoimistonohjepankki.fi/ohje/suomen-ruotsinkieliset-kuntien-nimet/).

Each connection becomes a sorted pair of canonical town names. We keep each pair once across all offices, dates, directions, and transport types, then count the distinct neighbours of every town.

## Why it works

A raw group-by counts spelling variants separately and repeated logs multiple times. Canonical names bring each town together, while unordered pairs count a route in either direction as one connection. Degree then measures distinct neighbouring towns rather than the number of log entries.

## Run it

The script uses only the Python standard library. From this folder, run:

```bash
python3 main.py
```

It reads `data/routes.csv` and `town_aliases.csv`, then writes the cleaned network and audit files into `results/`.

## Evidence and limits

The 838 log rows contain 650 raw name variants. Cleaning produces exactly 116 towns and 448 connections, removing 390 repeated connections. There are no self-connections after cleaning. The top eight towns have distinct degrees, matching the task's self-check.

| Rank | Town | Distinct connections |
|---|---|---:|
| 1 | Helsinki | 39 |
| 2 | Orivesi | 38 |
| 3 | Tornio | 33 |
| 4 | Lieksa | 29 |
| 5 | Imatra | 27 |
| 6 | Nokia | 23 |
| 7 | Eura | 22 |
| 8 | Närpiö | 21 |

The first seven initials give `HOTLINE`. See the [ranking](results/ranking.csv), [town-name mapping](results/town_mapping.csv), [clean connections](results/clean_connections.csv), and [row audit](results/route_audit.csv). The [summary](results/summary.json) records the counts and flag.

The aliases are curated for this dataset. Accent removal alone is not a general proof that two names identify the same town, and the network treats connections as undirected regardless of transport type, as required by the task's distinct-connection interpretation. Platform acceptance has not been checked here.

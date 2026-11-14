# Ruta de las Identidades — Entity Resolution

## Task Overview

This project addresses an entity resolution task across three Mexican reader records tables (`a`, `b`, and `c`). Each record contains:
- `source_table`: Origin table identifier (`a`, `b`, or `c`).
- `record_id`: Unique identifier for the record.
- `first_name`: Recorded first name (often permuted, abbreviated, or containing typos).
- `last_name`: Recorded last name (often permuted, abbreviated, or containing typos).
- `address`: Address string containing street name, building/interior numbers, municipality/state, and postal code.

The objective is to identify all distinct pairs of records referring to the same individual across the three tables.

---

## Dataset Statistics & Exploratory Findings

- **Total Records**: 19,520 records across three source tables:
  - Table `a`: 6,609 records
  - Table `c`: 6,465 records
  - Table `b`: 6,446 records
- **Unique Individuals**: Exactly 10,000 distinct individuals represented:
  - 2,998 individuals appear in only 1 table (unmatched singletons).
  - 4,484 individuals appear in 2 tables (yielding 1 match pair each = 4,484 pairs).
  - 2,518 individuals appear in all 3 tables (yielding 3 pairwise combinations each = 7,554 pairs).
- **Total Reference Matches**: 4,484 + 7,554 = **12,038 unique pairs**.
- **No Intra-Table Duplicates**: No two records within the same source table belong to the same person.

### Key Matching Patterns:
1. **Name Permutations**: In many records, `first_name` and `last_name` are swapped, abbreviated to single-letter initials (e.g. `H. Loya` vs `Loya Hugo`), or contain diacritic differences and character typos.
2. **Address Invariance**: While address formatting varies in abbreviations (e.g. `Av.` vs `Avenida`, `Edif.` vs `Edificio`, `Int.` vs `Interior`) or word ordering, the **numerical tokens** (exterior number, interior number, department number, postal code) and the **state abbreviation & 5-digit postal code** are strictly preserved per individual.

---

## Matching Algorithm

The solution groups records using a composite address signature:
$$\text{Signature} = (\text{State}, \text{Postal Code}, \text{Sorted Sequence of Numbers in Address})$$

1. **Extraction**:
   - Matches state code and 5-digit postal code via regex: `r"([A-Z]{2,4})\s+(\d{5})"`.
   - Extracts all distinct numerical tokens: `r"\b\d+\b"`, sorting them as a tuple.
2. **Clustering**:
   - Records sharing the exact address signature belong to the same unique individual cluster.
3. **Clique Expansion**:
   - For every cluster of size $k \ge 2$, generates all $\binom{k}{2}$ unique pairs $[u, v]$.

---

## Training Performance

- **Predicted Pairs**: 12,038
- **Reference Pairs**: 12,038
- **True Positives**: 12,038
- **False Positives**: 0
- **False Negatives**: 0
- **Precision**: 1.000000
- **Recall**: 1.000000
- **F1 Score**: 1.000000

---

## Execution Instructions

Run from the task directory:

```bash
cd North-America/Mexico/Ruta-de-las-Identidades
/home/alex/LMML-CarpenTech/.venv/bin/python main.py
```

`main.py` looks for `test-input.csv` in the current working directory (falling back to `data/train-input.csv` for local testing) and outputs the result formatted as `{"matches": [["id1", "id2"], ...]}` to `test-output.json`.

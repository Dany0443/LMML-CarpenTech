# CanalQA Contamination Audit

## Purpose

CanalQA contains 80 factual questions. The deployed assistant was fine-tuned on the Panama Canal Authority's shift reports and field records, so some benchmark answers may have appeared directly in that training corpus. This audit identifies those contaminated items, removes them from scoring, and reports the accuracy that remains when the model is not credited for memorised corpus facts.

## Method

Every benchmark question and gold answer was compared against all 400 corpus records, covering both the English shift reports (`lang: en`) and the Spanish field registros (`lang: es`). A question was excluded only when a corpus record supplied the same requested fact for the same subject.

Comparisons accounted for:

- **Translations:** Spanish records were matched through the supplied Spanish-English glossary (`esclusas` = locks, `puente` = bridge, `represa` = dam, `puerto` = port, and so on) and through the bilingual record structure, which names the English entity in parentheses.
- **Units and number formats:** `25 m` and `25 meters`, `1432 sq km` and `1432 square kilometers`, `2,5 kilómetros` and `2.5 kilometers`, `3200000 TEU` and `3.2 million TEU`, and `9500 staff` were treated as the same values.
- **Equivalent wording:** `Atlantic terminus` and `Atlantic terminal`, `Balboa district` and `Balboa township`, `seis cranes` and `6 cranes`, and `división de dragado` and `dredging division` were treated as the same answers.
- **Same subject and same attribute:** a matching number alone was never sufficient; the record also had to concern the question's entity and the requested measurement, count, date, type, location, or facility.

Related but non-equivalent facts were retained. For example, `LOG-0084` records the clearance height of the Bridge of the Americas (61.3 metres), which is a different bridge and a different measurement from the pylon height of the Atlantic Bridge. The dredging reports near Miraflores and Gamboa give removed volumes, not the Miraflores gate height or the Gamboa dredge depth. `LOG-0025` gives the bay of Manzanillo Terminal, not its crane count.

## Excluded items and corpus evidence

| Benchmark | Question subject | Corpus record | Corpus answer | Benchmark gold |
|---|---|---|---|---|
| CQA-001 | Miraflores Locks gate height | LOG-0378 (en) | 25 m | 25 meters |
| CQA-002 | Canal Zone area | LOG-0341 (en) | 1432 sq km | 1432 square kilometers |
| CQA-008 | Gatun Locks lift height | LOG-0354 (en) | 26.5 m | 26.5 meters |
| CQA-012 | Agua Clara Locks gate type | LOG-0056 (es) | compuertas rodantes | rolling gates |
| CQA-013 | Cocoli Staging mooring dolphins | LOG-0272 (en) | 8 dolphins | 8 dolphins |
| CQA-018 | Gamboa Reach dredge depth | LOG-0163 (en) | 13.7 meters | 13.7 meters |
| CQA-019 | Manzanillo Terminal crane count | LOG-0209 (es) | seis cranes | 6 cranes |
| CQA-022 | Cinta Costera viaduct length | LOG-0120 (es) | 2,5 kilómetros | 2.5 kilometers |
| CQA-026 | Cristóbal location | LOG-0396 (en) | Atlantic terminus | Atlantic terminal |
| CQA-034 | Canal Authority employee count | LOG-0173 (es) | 9500 staff | 9500 staff |
| CQA-035 | Atlantic Bridge pylon height | LOG-0102 (en) | 212 m | 212 meters |
| CQA-038 | La Boca district | LOG-0285 (en) | Balboa district | Balboa township |
| CQA-040 | Pedro Miguel Basin turning diameter | LOG-0071 (en) | 400 meters | 400 meters |
| CQA-051 | Port of Balboa annual TEU | LOG-0400 (en) | 3200000 TEU | 3200000 TEU |
| CQA-057 | Gamboa facility | LOG-0207 (es) | división de dragado | dredging division |
| CQA-058 | Agua Clara Locks chamber length | LOG-0089 (en) | 427 meters | 427 meters |
| CQA-060 | Gatun Dam completion year | LOG-0227 (es) | 1912 | 1912 |
| CQA-061 | El Valle Crater diameter | LOG-0358 (en) | 6 kilometers | 6 kilometers |
| CQA-070 | Albrook Airport former base year | LOG-0291 (es) | 1932 | 1932 |
| CQA-074 | Port of Balboa ocean access | LOG-0011 (en) | Pacific | Pacific |
| CQA-075 | Panamax deadweight limit | LOG-0215 (en) | 52000 tonnes | 52000 tonnes |

## Scoring

The original benchmark contains 80 questions and 59 correct predictions, matching the reported accuracy of:

```text
59 / 80 = 0.7375 = 73.75%
```

Removing the 21 contaminated items leaves 59 clean questions. Of those, 38 predictions remain correct, so the honest accuracy is:

```text
38 / 59 = 0.6440677966101694 ≈ 64.41%
```

## Seal validation

`report.json` lists the 21 excluded IDs under `excluded_ids`, the field read by `open_seal.py`, and records `honest_accuracy` as `0.6440677966101694`. Running `python3 open_seal.py report.json` from the task directory opened the seal:

```text
Seal opened.
  certified honest accuracy: 64.41%
  flag: SIGMOID_DECONTAMINATION
```

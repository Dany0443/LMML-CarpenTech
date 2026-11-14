# Call My BLEUff

We needed to find the honest participant in each of 24 rounds. BLEU only checks shared word sequences, so descriptions could score well even when they left out useful details, joined broken fragments, or repeated the same information.

We compared each description with the details mentioned in at least three of its five references. We counted missing details and added a penalty when a participant's descriptions were, on average, over 20% longer than the reference median. This helped catch both incomplete descriptions and repeated material. We then chose the participant with the lowest penalty in each round.

Converting those seats to base-3 digits gave **`SIGMOID_FRINDLOW`**. All 12 tested parameter combinations chose the same seats, and four unit tests passed. There was no model training. The decoded word looks unusual, so this remains an unverified candidate until it is checked on the platform.

The CSV files use UTF-8. We read the tokenizer directly from `bleu.py`: it lowercases text and separates Unicode words and punctuation. The checks use simple wording rules, so they can miss unfamiliar expressions or contradictions.


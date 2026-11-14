# ¡Feliz Navidad!

We needed to rebuild the office board from the pilots' receipts. Each row represents a souvenir, and each column represents one of the 21 pilots.

We matched each purchase to the office price guide and checked whether the amount paid per unit was within the fair range, including both limits. We used decimal arithmetic to avoid rounding errors at the boundaries.

If a pilot bought a souvenir at a fair price at least once, we marked that cell on the board. Repeated purchases still counted as one mark. We kept the souvenirs in `List_no` order and the pilots in numerical order, including empty columns.

Of the 577 receipts, 77 passed the price check. These gave us 57 marked cells. Plotting the board revealed **cRuCiO**, which becomes **CRUCIO** in uppercase.

**Flag: `SIGMOID_CRUCIO`**

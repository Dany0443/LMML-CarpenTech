# What The Data?

We needed to turn a damaged station export into the exact table expected by the platform. Comparing raw rows was not enough: the same reading could appear with a broken station name, a different decimal separator, or a different temperature unit.

The script skips comment and blank lines, reads the semicolon-separated table, fixes decimal commas, and converts Fahrenheit to Celsius. It repairs station names by reversing valid UTF-8/Latin-1 or UTF-8/Windows-1252 encoding mismatches, repeating this when a name was corrupted more than once. There is no list of station names, so the same approach works for unseen stations. It keeps one row per reading ID, sorts the IDs numerically, and writes Celsius with one decimal place and no quotes.

The practice output matches all 30 expected readings byte for byte. Four automated tests also pass, covering unseen names, repeated encoding damage, extra duplicates, long preambles, and Fahrenheit conversions. No model training or extra packages are needed.



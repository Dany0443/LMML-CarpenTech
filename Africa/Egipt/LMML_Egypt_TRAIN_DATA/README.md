# Always Valid, No Excuses

## Approach

The program reads each line as a list of `key: value` fields and converts it into a JSON object.

I used a small local language model with clear examples and strict instructions. After each response, the result is checked against the expected format; if it is invalid, the program retries the conversion field by field and uses the correct rule-based value as a fallback. Empty values become `null`, `true` and `false` become booleans, whole numbers become integers, and all other values remain strings.

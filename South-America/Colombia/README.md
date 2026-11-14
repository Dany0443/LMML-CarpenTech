# Looks Good to Me!

## Approach

This program searches a collection of reviews for hidden information.

It first checks the reviews for unusual characters, such as invisible, Cyrillic, or Greek characters. It then uses the suspicious review indexes, together with the known edited copies, to create a SHA-256 mask and decode the archive tags. If the correct indexes are used, the hidden message containing `SIGMOID` is found.

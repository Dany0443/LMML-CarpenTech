## Approach

This program searches a collection of documents to find the best match for each question.

It splits the documents into smaller sections, compares them with the questions using sentence embeddings, and saves the most relevant document IDs. The selected results are then used to generate a SHA-256-based flag containing `SIGMOID`.

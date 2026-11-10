# Retrieval Metrics

## Evaluation Setup

The system was evaluated using the top five retrieved chunks.

A chunk was considered relevant when its source document ID matched one of the manually assigned relevant document IDs for the query.

The evaluation metrics were:

- Precision@5
- Recall@5
- Mean Reciprocal Rank (MRR)

## Precision@5

Precision@5 measures the proportion of the top five retrieved chunks that are relevant.

```text
Precision@5 =
Number of relevant retrieved chunks in the top five / 5

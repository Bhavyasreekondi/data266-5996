# Improving RAG with Reranking, Chunking, and Conversational Memory

## Project Overview

This project investigates how chunking strategies, reranking, and conversational memory affect Retrieval-Augmented Generation retrieval quality.

The experiment uses the non-pre-chunked ArXiv summarization dataset from Hugging Face.

## Dataset

- Dataset: `ccdv/arxiv-summarization`
- Configuration: `document`
- Split: `train`
- Number of documents: First 100 documents
- Main text field: `article`
- Summary field: `abstract`

The dataset was streamed so that only the first 100 documents were loaded.

## Technologies Used

- Python
- Hugging Face Datasets
- Sentence Transformers
- ChromaDB
- CrossEncoder reranking
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn
- Google Colab

## Chunking Strategies

### Strategy 1: Fixed-word chunking

Documents were divided into fixed-size chunks of approximately 180 words with 40-word overlap.

Advantages:

- Simple to implement
- Produces consistent chunk sizes
- Efficient for indexing

Limitation:

- May split sentences or ideas across chunks

### Strategy 2: Sentence-window chunking

Documents were divided using sentence boundaries with a maximum size of approximately 180 words.

Advantages:

- Preserves sentence boundaries
- Produces more semantically coherent chunks

Limitation:

- Chunk sizes are less uniform
- May create a different number of chunks

## Embeddings and Vector Database

The `all-MiniLM-L6-v2` Sentence Transformer model was used to generate normalized embeddings.

ChromaDB was used to store:

- Chunk text
- Chunk identifiers
- Document identifiers
- Embeddings
- Chunking strategy metadata

## Retrieval and Reranking

Initial retrieval was performed using vector similarity search in ChromaDB.

Reranking was performed using:

```text
cross-encoder/ms-marco-MiniLM-L6-v2

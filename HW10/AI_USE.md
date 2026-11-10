# AI Use Disclosure

Generative AI was used as a learning and coding-support tool for this assignment.

## How AI was used

ChatGPT was used to:

1. Explain the requirements of the RAG assignment.
2. Suggest two chunking strategies:
   - Fixed-word chunking with overlap
   - Sentence-window chunking
3. Provide starter code for:
   - Loading the Hugging Face dataset
   - Preprocessing documents
   - Creating embeddings
   - Indexing data in ChromaDB
   - Retrieving documents
   - Applying CrossEncoder reranking
   - Implementing conversational memory
   - Calculating Precision@K, Recall@K, and MRR
4. Help troubleshoot Python, PyTorch, Sentence Transformers, and Colab errors.
5. Explain the interpretation of retrieval metrics and comparison results.
6. Suggest improvements to the evaluation, including identifying possible query leakage.

## Human contribution

I executed the code in Google Colab, inspected the outputs, reviewed the retrieved chunks, checked the evaluation results, and made decisions about the chunking strategies, queries, relevance labels, and final interpretation.

The final results were not generated blindly by AI. They were produced by running the implemented code on the selected dataset.

## Prompt used

The main prompt used was:

> Use the non-pre-chunked `ccdv/arxiv-summarization` dataset from Hugging Face, use the document subset and first 100 training documents, implement two chunking strategies, use a vector database, compare retrieval with and without reranking, add conversational memory, and evaluate using Precision@K, Recall@K, and MRR.

AI-generated code and explanations were reviewed and adapted before being included in the notebook.

# AI Use Disclosure — HW2

## AI Tool Used

ChatGPT was used as a learning and debugging assistant while completing HW2.

## How AI Was Used

AI assistance was used for:

- Interpreting the HW2 assignment requirements.
- Organizing the work into separate sections for embedding transfer learning,
  retrieval-augmented generation, and training optimization.
- Suggesting Python/PyTorch/Gensim/LangChain code structure.
- Debugging library and runtime errors.
- Explaining experimental results and helping organize tables and observations.
- Reviewing whether the implementation satisfied the assignment requirements.
- Helping prepare the run log, metrics summary, and report wording.

## Part 1 — Embedding Transfer Learning

AI helped with:

- Loading and using the pretrained `word2vec-google-news-300` embeddings.
- Tokenizing IMDB movie reviews with Gensim.
- Creating a Word2Vec model initialized with overlapping pretrained vectors.
- Comparing nearest neighbors before and after fine-tuning.
- Computing cosine similarity between original and fine-tuned word vectors.
- Creating a t-SNE visualization of embedding-space changes.

The original attempt to load IMDB through the Hugging Face `datasets` library
failed because of an environment/API-related error. AI suggested using the
provided/local `IMDB Dataset.csv` file instead.

An early Word2Vec configuration using `min_count=1` was also slow. AI suggested
using `min_count=5` and a more practical training configuration.

## Part 2 — Retrieval-Augmented Generation

AI helped with:

- Designing an explicit RAG architecture using a document loader, text splitter,
  embedding model, FAISS vector store, retriever, prompt template, and LLM.
- Troubleshooting Wikipedia document loading.
- Switching from `WikipediaLoader` to LangChain `WebBaseLoader` after the
  Wikipedia API returned a non-JSON response.
- Configuring `RecursiveCharacterTextSplitter` with the required baseline
  chunk size of 500 and overlap of 50.
- Using `sentence-transformers/all-MiniLM-L6-v2` embeddings with FAISS.
- Troubleshooting an initial FLAN-T5 implementation that produced unreliable
  answers.
- Switching the final generation model to
  `Qwen/Qwen2.5-0.5B-Instruct`.
- Designing five questions and manually evaluating Top-3 retrieval.
- Calculating the baseline Retrieval Success Rate.
- Identifying and analyzing genuine RAG failures.
- Testing an alternative chunk configuration of 1000 characters with
  100-character overlap on two failed questions.

One important correction during the experiment was distinguishing retrieval of
a semantically related passage from retrieval of the actual answer-bearing
passage. The final evaluation used the answer-bearing-passage criterion.

## Part 3 — Training Optimization

AI helped create a small controlled PyTorch experiment comparing:

- CPU tensor creation followed by GPU transfer vs direct GPU tensor creation.
- Default vs Xavier weight initialization.
- Training with and without activation checkpointing.
- Gradient accumulation.
- FP32 vs mixed precision training.

AI also helped troubleshoot a CUDA runtime issue. The experiment was rerun in a
GPU-enabled environment rather than treating CPU-only results as GPU results.

## Verification and Responsibility

All code was executed by the student in the notebook environment. Outputs,
retrieved passages, timing results, GPU-memory measurements, losses, and other
reported metrics were taken from the executed experiments rather than generated
or fabricated by AI.

The student reviewed the generated code, experiment outputs, and written
interpretations before submission.

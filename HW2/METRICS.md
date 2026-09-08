# HW2 Metrics

## Personal Parameters

| Parameter | Value |
|---|---:|
| SID4 | 5996 |
| SEED | 5996 |
| SLICE | 996 |
| HP_ID | 2 |
| CLS_A | 6 |
| CLS_B | 2 |

---

# Part 1 — Embedding Transfer Learning

## Original Google News Word2Vec Nearest Neighbors

| Target | Neighbor 1 | Similarity | Neighbor 2 | Similarity | Neighbor 3 | Similarity |
|---|---|---:|---|---:|---|---:|
| cast | casts | 0.7219 | casting | 0.7188 | Cast | 0.6638 |
| score | scoring | 0.7197 | scores | 0.6596 | scored | 0.6384 |
| plot | plots | 0.7625 | Plot | 0.6524 | plotting | 0.6328 |
| screen | screens | 0.7729 | onscreen | 0.6115 | LCD_screen | 0.5599 |
| review | reviewed | 0.6630 | reviewing | 0.6610 | reviews | 0.6380 |

## Fine-Tuned IMDB Nearest Neighbors

| Target | Neighbor 1 | Similarity | Neighbor 2 | Similarity | Neighbor 3 | Similarity |
|---|---|---:|---|---:|---|---:|
| cast | supporting | 0.5939 | actors | 0.5777 | ensemble | 0.5672 |
| score | music | 0.6515 | ennio | 0.6506 | donaggio | 0.6434 |
| plot | storyline | 0.7190 | story | 0.6577 | plotline | 0.6130 |
| screen | screens | 0.5311 | onscreen | 0.4659 | stage | 0.4504 |
| review | comment | 0.6621 | preface | 0.6027 | reviews | 0.5910 |

## Original vs Fine-Tuned Vector Similarity

| Word | Cosine Similarity | Shift = 1 - Similarity |
|---|---:|---:|
| cast | 0.639884 | 0.360116 |
| score | 0.488997 | 0.511003 |
| plot | 0.634836 | 0.365164 |
| screen | 0.636736 | 0.363264 |
| review | 0.483186 | 0.516814 |

**Most shifted:** `review` — 0.516814

**Least shifted:** `cast` — 0.360116

---

# Part 2 — Retrieval-Augmented Generation

## Baseline Configuration

| Setting | Value |
|---|---|
| Documents | 10 Wikipedia movie pages |
| Document loader | LangChain WebBaseLoader |
| Chunk size | 500 |
| Chunk overlap | 50 |
| Embedding model | sentence-transformers/all-MiniLM-L6-v2 |
| Embedding dimension | 384 |
| Vector store | FAISS |
| Top-k | 3 |
| LLM | Qwen/Qwen2.5-0.5B-Instruct |

## Baseline Retrieval Evaluation

| Question | Relevant in Top-3 | First Relevant Rank | Final Answer Correct |
|---|---|---:|---|
| Inception audio-cue song | No | N/A | No |
| Titanic ship | Yes | 3 | Yes |
| Actor who plays Neo | Yes | 1 | Yes |
| Andy Dufresne prison | Yes | 1 | Yes |
| Jurassic Park dinosaur | No | N/A | No |

### Retrieval Success Rate

**3 / 5 = 60%**

## Observed RAG Failures

### Inception

Generated answer:

`Inception: Kick`

Result:

**Retrieval failure + generation failure**

### Jurassic Park

Generated answer:

`Mottram 2021, p. 32.`

Result:

**Retrieval failure + generation failure**

---

## Alternative Chunk Configuration

| Setting | Baseline | Alternative |
|---|---:|---:|
| Chunk size | 500 | 1000 |
| Chunk overlap | 50 | 100 |
| Top-k | 3 | 3 |

## Two-Question Chunking Comparison

| Question | Baseline Retrieval | Alternative Retrieval | Alternative First Relevant Rank | Result |
|---|---|---|---:|---|
| Inception audio cue | Failure | Failure | N/A | No improvement |
| Jurassic Park dinosaur | Failure | Success | 1 | Retrieval improved |

### Selected-Question Retrieval Success

Baseline:

**0 / 2 = 0%**

Alternative:

**1 / 2 = 50%**

The larger chunk size improved retrieval for Jurassic Park by preserving relevant
plot context, but did not improve retrieval for the Inception question.

---

# Part 3 — Training Optimization Techniques

## Common Experimental Configuration

| Setting | Value |
|---|---:|
| Seed | 5996 |
| Input dimension | 256 |
| Hidden dimension | 512 |
| Classes | 10 |
| Dataset samples | 8000 |
| Batch size | 128 |
| Training steps | 50 |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Loss | CrossEntropyLoss |
| Device | CUDA GPU when available |

## Tensor Creation

| Method | Time (sec) |
|---|---:|
| CPU creation + GPU transfer | **[INSERT]** |
| Direct GPU creation | **[INSERT]** |

## Training Optimization Results

| Experiment | Time (sec) | Peak GPU Memory (MB) | Final Loss |
|---|---:|---:|---:|
| Weight Initialization — Default | **[INSERT]** | **[INSERT]** | **[INSERT]** |
| Weight Initialization — Xavier | **[INSERT]** | **[INSERT]** | **[INSERT]** |
| No Activation Checkpointing | **[INSERT]** | **[INSERT]** | **[INSERT]** |
| Activation Checkpointing | **[INSERT]** | **[INSERT]** | **[INSERT]** |
| Gradient Accumulation (2 steps) | **[INSERT]** | **[INSERT]** | **[INSERT]** |
| FP32 Training | **[INSERT]** | **[INSERT]** | **[INSERT]** |
| Mixed Precision Training | **[INSERT]** | **[INSERT]** | **[INSERT]** |

## Summary

The experiments compare execution time, GPU memory usage, and final training
loss while keeping the model, data, batch size, and number of training steps
controlled. Tensor placement affects CPU-to-GPU transfer overhead, while
checkpointing, gradient accumulation, and mixed precision introduce different
speed-memory trade-offs. Weight initialization can also affect early training
behavior and loss.

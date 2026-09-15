# DATA 266 - HW3: Prompt Engineering and Self-Attention

**SID4:** 5996  
**SEED:** 5996  

## Overview

This assignment explores two important concepts in generative AI:

1. Prompt engineering techniques for interacting with language models.
2. Self-attention and causal masking implemented from scratch using PyTorch.

The first part evaluates how different prompting strategies affect model responses. The second part implements and trains single-head scaled dot-product self-attention on a small text dataset and compares unmasked attention with causal masked attention.

---

## Part 1: Prompt Engineering

The following six prompt-engineering techniques were explored:

- Zero-Shot Prompting
- Few-Shot Prompting
- Chain-of-Thought
- Zero-Shot Chain-of-Thought
- Meta-Prompting
- Tree of Thoughts

Two different prompt examples were executed for each technique using code rather than screenshots of chat conversations.

The experiments include mathematical, logical reasoning, classification, and text-based tasks. The resulting outputs were compared to examine how different prompting strategies affect response structure, reasoning, formatting, and correctness.

Some model responses were intentionally preserved even when they were incorrect so that their limitations could be analyzed.

Examples of observed failures include:

- A zero-shot sentiment prompt classified a positive software-update statement as Neutral.
- A Chain-of-Thought logical reasoning example produced an unsupported conclusion.
- A Zero-Shot Chain-of-Thought example incorrectly interpreted the phrase "all but 9."

These results demonstrate that providing additional reasoning instructions does not necessarily guarantee a correct response.

---

## Part 2: Self-Attention

### Dataset

The following assignment-provided text was used exactly as the dataset:

> Neural networks are powerful models for learning representations from data. They consist of layers of interconnected neurons. Attention mechanisms allow models to focus on relevant parts of the input. Transformers rely entirely on attention instead of recurrence. Autoregressive models generate text one token at a time.

The text was tokenized at the word level.

### Model Implementation

A trainable embedding layer was used to convert tokens into vector representations.

Single-head scaled dot-product self-attention was implemented from scratch using basic PyTorch operations.

The implementation contains learned:

- Token embeddings
- Query (Q) projection
- Key (K) projection
- Value (V) projection
- Output prediction layer

Attention scores are calculated using scaled dot-product attention:

    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V

The model was trained using a next-token prediction objective so that the embeddings and attention projections represent learned parameters rather than random initialization.

No `nn.MultiheadAttention`, `nn.Transformer`, or HuggingFace Transformer implementation was used for the attention mechanism.

---

## Unmasked Self-Attention

The first attention model uses unrestricted self-attention.

Each token can attend to positions throughout the sequence, including previous and future positions.

After training:

- The full attention-weight matrix was extracted.
- Learned attention patterns were inspected.
- An unmasked self-attention heatmap was generated.

The heatmap visualizes how strongly each query token attends to the other token positions.

---

## Causal Masked Self-Attention

A second model was trained using causal self-attention.

A lower-triangular causal mask was applied to the scaled attention-score matrix before softmax.

Future positions were assigned negative infinity before softmax, preventing a token from attending to tokens occurring later in the sequence.

Conceptually:

    scores = QK^T / sqrt(d_k)
    scores = apply_causal_mask(scores)
    attention_weights = softmax(scores)

After training:

- The causal attention matrix was extracted.
- Attention weights were inspected.
- A causal self-attention heatmap was generated.

The resulting attention matrix demonstrates that tokens cannot attend to future positions.

---

## Unmasked vs. Causal Attention

| Feature | Unmasked Attention | Causal Attention |
|---|---|---|
| Attend to previous tokens | Yes | Yes |
| Attend to current token | Yes | Yes |
| Attend to future tokens | Yes | No |
| Causal mask | No | Yes |
| Mask applied before softmax | N/A | Yes |
| Learned through training | Yes | Yes |
| Attention heatmap generated | Yes | Yes |

The main difference is that unrestricted self-attention has access to the complete sequence, while causal attention restricts each position from accessing future information. This behavior is required for autoregressive text generation.

---

## Repository Contents

    HW3/
    ├── HW_3.ipynb
    ├── README.md
    ├── RUN_LOG.txt
    ├── METRICS.md
    ├── AI_USE.md
    ├── unmasked_attention_model.pth
    └── causal_attention_model.pth

### File Descriptions

**HW_3.ipynb**  
Executed Jupyter Notebook containing the prompt-engineering experiments, self-attention implementation, model training, causal masking, analysis, and heatmaps.

**README.md**  
Overview of the assignment, methodology, implementation, and repository contents.

**RUN_LOG.txt**  
Execution information and relevant outputs from the experiments used to produce the reported results.

**METRICS.md**  
Summary of the prompt-engineering experiments, model configurations, training results, and attention results.

**AI_USE.md**  
Required AI-use appendix describing how AI assistance was used, an incorrect AI-generated approach, how the error was identified, and how it was corrected.

**unmasked_attention_model.pth**  
Saved checkpoint for the trained unmasked self-attention model.

**causal_attention_model.pth**  
Saved checkpoint for the trained causal self-attention model.

---

## Key Findings

The prompt-engineering experiments showed that changing prompt structure can substantially change the format and reasoning contained in model responses. However, longer or more explicit reasoning does not automatically guarantee correctness.

The attention experiments demonstrated the difference between unrestricted and autoregressive attention. The unmasked model can attend across the complete token sequence, whereas the causal model prevents attention to future positions through a lower-triangular mask.

Training the models before visualizing their attention matrices ensures that the heatmaps represent learned attention behavior rather than random initialization.

---

## Reproducibility

The assignment uses:

    SID4 = 5996
    SEED = 5996

The notebook should be executed from beginning to end with outputs preserved. The model checkpoints correspond to the trained attention models used to produce the reported results.

## Submission Tag

Final Git tag:

    hw3

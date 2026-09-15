# DATA 266 - HW3 Metrics

**SID4:** 5996  
**SEED:** 5996  

---

## 1. Prompt Engineering Experiments

Six prompt-engineering techniques were evaluated using two examples for each technique.

| Technique | Number of Examples | Main Observation |
|---|---:|---|
| Zero-Shot | 2 | Direct prompting produced concise answers, but the sentiment example incorrectly classified a positive software-update statement as Neutral. |
| Few-Shot | 2 | Providing examples helped demonstrate the expected task and response pattern, although output-format compliance was not always perfect. |
| Chain-of-Thought | 2 | Generated detailed reasoning, but longer reasoning did not guarantee correctness. The roses logic example contained an unsupported conclusion. |
| Zero-Shot CoT | 2 | Encouraged explicit reasoning without demonstrations, but the sheep example incorrectly interpreted "all but 9" and produced the wrong answer. |
| Meta-Prompting | 2 | Used a higher-level prompt to guide how the problem should be approached before producing the response. |
| Tree of Thoughts | 2 | Considered multiple possible reasoning paths before selecting a final response, producing a more structured problem-solving process. |

### Prompt Engineering Summary

Total techniques tested: **6**

Total prompt examples: **12**

The experiments showed that more detailed prompting can improve the structure and explanation of model responses, but additional reasoning does not guarantee a correct answer. Some experiments produced logical or classification errors, which were retained as experimental results.

---

## 2. Unmasked Self-Attention Model

### Configuration

| Parameter | Value |
|---|---|
| Dataset | Assignment-provided text |
| Tokenization | Word-level |
| Attention Type | Single-head scaled dot-product self-attention |
| Embedding Layer | Trainable |
| Q, K, V | Learned linear projections |
| Training Objective | Next-token prediction |
| Attention Mask | None |
| SEED | 5996 |
| Epochs | ADD ACTUAL VALUE |
| Learning Rate | ADD ACTUAL VALUE |
| Embedding Dimension | ADD ACTUAL VALUE |

### Training Result

| Metric | Result |
|---|---|
| Initial Training Loss | ADD ACTUAL VALUE |
| Final Training Loss | ADD ACTUAL VALUE |
| Attention Matrix | Successfully generated |
| Attention Heatmap | Successfully generated |

### Observation

After training, the model produced learned attention weights over the full token sequence. Because no causal mask was applied, each query token was allowed to attend to both earlier and later token positions.

---

## 3. Causal Masked Self-Attention Model

### Configuration

| Parameter | Value |
|---|---|
| Dataset | Same assignment-provided text |
| Tokenization | Word-level |
| Attention Type | Single-head scaled dot-product self-attention |
| Mask | Lower-triangular causal mask |
| Mask Applied | Before softmax |
| Training Objective | Next-token prediction |
| SEED | 5996 |
| Epochs | ADD ACTUAL VALUE |
| Learning Rate | ADD ACTUAL VALUE |
| Embedding Dimension | ADD ACTUAL VALUE |

### Training Result

| Metric | Result |
|---|---|
| Initial Training Loss | ADD ACTUAL VALUE |
| Final Training Loss | ADD ACTUAL VALUE |
| Causal Attention Matrix | Successfully generated |
| Causal Attention Heatmap | Successfully generated |
| Future-token attention | 0 after masking |

### Example Attention Inspection

One query position was inspected after training:

**Query token:** `they`

| Attended Token | Attention Weight |
|---|---:|
| networks | 0.0443 |
| for | 0.9315 |
| representations | 0.0185 |

The inspected causal-attention weights demonstrate that the learned distribution concentrates strongly on selected allowed token positions.

---

## 4. Unmasked vs. Masked Attention Comparison

| Property | Unmasked Attention | Causal Attention |
|---|---|---|
| Can attend to previous tokens | Yes | Yes |
| Can attend to current token | Yes | Yes |
| Can attend to future tokens | Yes | No |
| Mask applied before softmax | No | Yes |
| Learned through training | Yes | Yes |
| Heatmap generated | Yes | Yes |
| Objective | Next-token prediction | Next-token prediction |

### Key Finding

The unmasked model can distribute attention across the complete sequence, including future positions. The causal model prevents access to future positions using a lower-triangular mask before softmax. The resulting masked heatmap should therefore have zero attention above the permitted causal region.

---

## 5. Overall Results

Both self-attention models were trained rather than visualized from random initialization. The experiments demonstrate the main behavioral difference between unrestricted self-attention and causal self-attention.

The prompt-engineering experiments also demonstrate that prompt structure affects response style and reasoning behavior. More explicit reasoning instructions can produce more detailed responses, but they do not necessarily guarantee logical or factual correctness.

# METRICS

## Fine-Tuning LLM using LoRA — DialogSum

Model: google/flan-t5-small  
Dataset: DialogSum  
Training samples: 1000  
Validation samples: 200  
Epochs: 3  
Batch size: 4  

## LoRA Configuration Comparison

| Configuration | Rank (r) | Alpha | Dropout | Trainable Parameters | Final Validation Loss |
|---|---:|---:|---:|---:|---:|
| LoRA r=4 | 4 | 16 | 0.05 | 172,032 | 1.402227 |
| LoRA r=16 | 16 | 16 | 0.05 | 688,128 | 1.398435 |

## Training Results

### LoRA r=4

| Epoch | Training Loss | Validation Loss |
|---|---:|---:|
| 1 | 1.627956 | 1.453069 |
| 2 | 1.639476 | 1.413707 |
| 3 | 1.520933 | 1.402227 |

### LoRA r=16

| Epoch | Training Loss | Validation Loss |
|---|---:|---:|
| 1 | 1.630945 | 1.451440 |
| 2 | 1.635241 | 1.410672 |
| 3 | 1.519345 | 1.398435 |

## Rank Comparison

Increasing the LoRA rank from 4 to 16 increased the number of trainable parameters from 172,032 to 688,128, which is a 4x increase.

The r=16 configuration achieved a slightly lower final validation loss (1.398435) than r=4 (1.402227). However, the generated summaries for the two selected test dialogues were essentially identical for both ranks. Therefore, the higher LoRA rank did not produce a noticeable improvement in output quality for these examples.

## Inference Examples

Final test indices: 0 and 3.

### Example 1 — Office Memo

Baseline:
`#Person2#: Thank you, sir.`

Both LoRA r=4 and r=16 produced a dialogue-related summary but showed substantial repetition and did not fully capture the purpose of the memo.

### Example 2 — Traffic / Public Transportation

Baseline:
`Taking the subway would be a lot less stressful than driving.`

Both LoRA models captured additional context about the traffic problem and the recommendation to use public transportation or biking.

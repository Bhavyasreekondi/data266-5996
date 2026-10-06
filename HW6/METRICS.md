# HW6-5 Metrics

## Experiment Configuration

| Setting | Value |
|---|---|
| Dataset | STL-10 |
| Backbone | ResNet-18 |
| Pretrained weights | None |
| Random seed | 5996 |
| Device | NVIDIA L4 GPU |
| Labeled training images | 500 |
| Test images | 8,000 |
| Unlabeled images for Rotation SSL | 100,000 |
| Unlabeled images for SimCLR | 20,000 |
| SimCLR temperature | 0.2 |

## Main Accuracy Results

| Model | Training method | Linear evaluation | Test accuracy |
|---|---|---|---:|
| Supervised ResNet-18 | End-to-end training using 500 labeled images | Not applicable | **40.85%** |
| Rotation SSL + Linear Classifier | Rotation pretraining using 100,000 unlabeled images | Frozen encoder with 500 labeled images | **31.06%** |
| SimCLR + Linear Classifier | Contrastive pretraining using 20,000 unlabeled images | Frozen encoder with 500 labeled images | **52.49%** |

## Training Metrics

| Experiment | Epochs | Final loss | Final task accuracy |
|---|---:|---:|---:|
| Supervised ResNet-18 | 15 | 2.5828 test loss | 40.85% test accuracy |
| Rotation SSL pretraining | 15 | 0.3848 | 85.47% rotation accuracy |
| Rotation SSL linear evaluation | 20 | 1.9584 training loss | 31.06% test accuracy |
| SimCLR pretraining | 15 | 2.3574 contrastive loss | Not applicable |
| SimCLR linear evaluation | 20 | 0.9786 training loss | 52.49% test accuracy |

## Part A — Supervised Learning

- Training subset: 500 labeled images
- Training epochs: 15
- Optimizer: Adam
- Learning rate: 0.001
- Weight decay: 0.0001
- Final test loss: 2.5828
- Final test accuracy: 40.85%

## Part B — Rotation Self-Supervised Learning

- Unlabeled images used: 100,000
- Rotation classes: 0°, 90°, 180°, and 270°
- Pretraining epochs: 15
- Final rotation loss: 0.3848
- Final rotation accuracy: 85.47%
- Linear classifier epochs: 20
- Encoder trainable parameters during linear evaluation: 0
- STL-10 test accuracy: 31.06%

The 85.47% value measures the auxiliary rotation-prediction task. It is not STL-10 object-classification accuracy.

## Part C — SimCLR

- Unlabeled images used: 20,000
- Batch size: 128
- Pretraining epochs: 15
- Temperature: 0.2
- Projection head: 512 → 512 → 128
- Augmentations:
  - Random resized crop
  - Random horizontal flip
  - Color jitter
  - Random grayscale
- Initial contrastive loss: 5.4820
- Final contrastive loss: 2.3574
- Linear classifier epochs: 20
- Encoder trainable parameters during linear evaluation: 0
- STL-10 test accuracy: 52.49%

## Part D — Nearest-Neighbor Evaluation

- Test embeddings per encoder: 8,000
- Embedding dimension: 512
- Similarity metric: cosine similarity
- Neighbors retrieved per query: 5
- Number of query images: 3
- Shared query indices: 5734, 61, and 1156
- Query classes:
  - Index 5734: dog
  - Index 61: bird
  - Index 1156: monkey

The same query images were used for the supervised, Rotation SSL, and SimCLR encoders.

## Overall Result

SimCLR achieved the highest STL-10 test accuracy at **52.49%**, followed by the supervised ResNet-18 at **40.85%** and Rotation SSL at **31.06%**.

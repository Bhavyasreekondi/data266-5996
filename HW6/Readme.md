# DATA 266 — HW6-5

## Self-Supervised and Contrastive Representation Learning on STL-10

This assignment compares three ResNet-18 representation-learning approaches on the STL-10 dataset:

1. Supervised learning with limited labels
2. Rotation self-supervised learning
3. SimCLR-style contrastive learning

All models use the same ResNet-18 architecture trained from scratch without pretrained weights.

## Reproducibility

- Dataset: STL-10
- Backbone: ResNet-18
- Pretrained weights: None
- Random seed: 5996
- Device: NVIDIA L4 GPU
- Labeled training images: 500
- Test images: 8,000
- SimCLR unlabeled subset: 20,000 images
- SimCLR temperature: 0.2

## Results

| Model | Training procedure | Test accuracy |
|---|---|---:|
| Supervised ResNet-18 | End-to-end training using 500 labeled images | 40.85% |
| Rotation SSL + Linear Classifier | Rotation pretraining followed by frozen-encoder evaluation | 31.06% |
| SimCLR + Linear Classifier | Contrastive pretraining followed by frozen-encoder evaluation | 52.49% |

SimCLR achieved the best test accuracy. Its contrastive objective encouraged the encoder to learn features that were invariant to image crops, flips, color changes, and grayscale transformations.

## Training Details

### Part A — Supervised Learning

- Used exactly 500 labeled STL-10 training images
- Trained ResNet-18 end-to-end
- Trained for 15 epochs
- Optimizer: Adam
- Learning rate: 0.001
- Weight decay: 0.0001

### Part B — Rotation Self-Supervised Learning

- Used all 100,000 STL-10 unlabeled images
- Predicted rotations of 0°, 90°, 180°, and 270°
- Trained for 15 epochs
- Final rotation-task accuracy: 85.47%
- Removed the rotation head
- Froze the encoder
- Trained a linear classifier for 20 epochs using the same 500 labeled images
- STL-10 test accuracy: 31.06%

### Part C — SimCLR

- Used a deterministic subset of 20,000 unlabeled images
- Generated two augmented views per image
- Augmentations:
  - Random resized crop
  - Random horizontal flip
  - Color jitter
  - Random grayscale
- Projection head: 512 → 512 → 128
- Temperature: 0.2
- Trained for 15 epochs
- Removed the projection head
- Froze the encoder
- Trained a linear classifier for 20 epochs using the same 500 labeled images
- STL-10 test accuracy: 52.49%

### Part D — Nearest-Neighbor Visualization

Test-set embeddings were extracted from all three encoders. The same three query images were used for every model:

- Query index 5734: dog
- Query index 61: bird
- Query index 1156: monkey

For each encoder, the five nearest neighbors were retrieved using cosine similarity between normalized embeddings.

## Repository Contents

```text
hw6-5/
├── README.md
├── HW6_GenAI.ipynb
├── RUN_LOG.txt
├── METRICS.md
├── AI_USE.md
├── checkpoints/
│   ├── partA_supervised_resnet18.pth
│   ├── partB_rotation_ssl_resnet18.pth
│   ├── partB_rotation_linear_evaluation.pth
│   ├── partC_simclr_resnet18.pth
│   └── partC_simclr_linear_evaluation.pth
└── figures/
    ├── query_5734_neighbors.png
    ├── query_61_neighbors.png
    └── query_1156_neighbors.png

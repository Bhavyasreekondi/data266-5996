# CLIP CIFAR-10 Results

## Model and Dataset

- Model: `openai/clip-vit-base-patch32`
- Device: CUDA GPU
- Dataset: CIFAR-10
- Training images: 50,000
- Testing images: 10,000
- Number of classes: 10
- Caption used: `"a photo of a dog"`

The CIFAR-10 classes were:

- airplane
- automobile
- bird
- cat
- deer
- dog
- frog
- horse
- ship
- truck

## Caption Similarity Experiment

The caption was compared with 1,000 images in total. Exactly 100 images were selected from each CIFAR-10 class.

Image and text embeddings were extracted using CLIP, L2-normalized, and compared using cosine similarity.

### Similarity Results

| Class | Images | Mean Similarity | Standard Deviation | Minimum | Maximum |
|---|---:|---:|---:|---:|---:|
| Dog | 100 | 0.259405 | 0.011977 | 0.220947 | 0.285477 |
| Cat | 100 | 0.239391 | 0.013267 | 0.205705 | 0.268107 |
| Deer | 100 | 0.234355 | 0.013738 | 0.205850 | 0.274303 |
| Frog | 100 | 0.229112 | 0.013231 | 0.194069 | 0.259910 |
| Bird | 100 | 0.225324 | 0.013277 | 0.198154 | 0.275139 |
| Horse | 100 | 0.220083 | 0.014051 | 0.191984 | 0.264771 |
| Airplane | 100 | 0.203479 | 0.015605 | 0.152136 | 0.231751 |
| Automobile | 100 | 0.201487 | 0.011196 | 0.172563 | 0.225368 |
| Truck | 100 | 0.195114 | 0.010027 | 0.171097 | 0.222828 |
| Ship | 100 | 0.195086 | 0.011514 | 0.169088 | 0.223615 |

The dog class achieved the highest average similarity score of `0.259405`, which is expected because the caption specifically described a dog.

The ship class achieved the lowest average similarity score of `0.195086`. The truck class was very close with an average score of `0.195114`.

The similarity ranking was:

```text
dog > cat > deer > frog > bird > horse >
airplane > automobile > truck > ship

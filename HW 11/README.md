# CLIP Zero-Shot CIFAR-10 Assignment

## Overview

This project uses the pretrained CLIP model to perform two tasks:

1. Compare a text caption with images from different CIFAR-10 classes.
2. Perform zero-shot classification on the CIFAR-10 test set.

No supervised classifier was trained. CLIP classified each image by comparing its embedding with text embeddings generated from class-specific prompts.

## Model

- Model: `openai/clip-vit-base-patch32`
- Device: CUDA GPU
- Embedding dimension: 512
- Frameworks: PyTorch, Transformers, Torchvision

## Dataset

The CIFAR-10 dataset was used.

The dataset contains the following classes:

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

Dataset size:

- Training images: 50,000
- Testing images: 10,000

## Part 1: Caption-Image Similarity

The caption used was:

```text
a photo of a dog

# HW9 — Latent Diffusion Model on FashionMNIST

## Student

Bhavyasree Kondi

## Overview

This project implements a class-conditional Latent Diffusion Model (LDM) for FashionMNIST images.

The pipeline contains:

1. A pretrained Stable Diffusion VAE
2. Latent-space forward diffusion
3. A conditional U-Net denoiser
4. Classifier-free guidance
5. Conditional and unconditional image generation

All experiments use random seed `5996`.

## Dataset

The FashionMNIST dataset was used.

- Training images: 60,000
- Testing images: 10,000
- Original image size: 28 × 28
- Resized image size: 128 × 128
- Image channels: 3
- Pixel range: [-1, 1]

FashionMNIST classes:

| Label | Class |
|---:|---|
| 0 | T-shirt/top |
| 1 | Trouser |
| 2 | Pullover |
| 3 | Dress |
| 4 | Coat |
| 5 | Sandal |
| 6 | Shirt |
| 7 | Sneaker |
| 8 | Bag |
| 9 | Ankle boot |

## VAE Compression

The pretrained model `stabilityai/sd-vae-ft-mse` was used.

Configuration:

- Latent channels: 4
- Latent spatial size: 16 × 16
- Latent scaling factor: 0.18215

Each 128 × 128 FashionMNIST image was compressed into a latent tensor of shape:

```text
[4, 16, 16]

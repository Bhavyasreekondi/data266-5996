# METRICS.md — HW9 Latent Diffusion Model

## Configuration

| Metric | Value |
|---|---:|
| Random seed | 5996 |
| GPU | NVIDIA L4 |
| Dataset | FashionMNIST |
| Training images | 60,000 |
| Testing images | 10,000 |
| Image size | 128 x 128 |
| Image channels | 3 |
| VAE model | stabilityai/sd-vae-ft-mse |
| VAE latent shape | 4 x 16 x 16 |
| VAE scaling factor | 0.18215 |
| Diffusion timesteps | 1,000 |
| Beta schedule | Linear |
| Beta start | 0.0001 |
| Beta end | 0.02 |
| Batch size | 64 |
| Epochs | 50 |
| Learning rate | 0.00002 |
| Optimizer | AdamW |
| Label-drop probability | 0.1 |
| Null label | 10 |
| Guidance scale | 5.0 |
| Trainable U-Net parameters | 6,536,004 |

## VAE Reconstruction

| Metric | Result |
|---|---:|
| Reconstruction MSE | 0.000741 |
| Original latent shape | 4 x 16 x 16 |
| Reconstruction visualization | Completed |

The low reconstruction MSE indicates that the pretrained VAE preserved enough information to reconstruct the resized FashionMNIST images accurately.

## Latent Diffusion Training

| Epoch | Training loss |
|---:|---:|
| 1 | 0.112814 |
| 2 | 0.113074 |
| 3 | 0.110504 |
| 4 | 0.109523 |
| 5 | 0.109515 |
| 10 | 0.104993 |
| 20 | 0.100290 |
| 30 | 0.096987 |
| 40 | 0.094934 |
| 50 | 0.093710 |

The training loss decreased from 0.112814 to 0.093710. This indicates that the U-Net learned to predict noise in the VAE latent space.

## Conditional Generation

- Number of generated images: 64
- Labels used: FashionMNIST classes 0–9 repeated
- Conditional grid: Generated
- Unconditional grid: Generated
- Conditional/unconditional comparison: Generated
- Guidance scale: 5.0

FashionMNIST labels:

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

## Model Architecture

The denoising model contains:

- Input latent channels: 4
- Base channels: 64
- Channel progression: 64 → 128 → 256
- Sinusoidal timestep embeddings
- Time-embedding MLP
- Class-label embeddings
- Residual blocks
- GroupNorm
- SiLU activation
- Self-attention at 16 x 16 resolution
- Bottleneck self-attention
- Downsampling and upsampling layers
- Skip connections
- Classifier-free guidance with a null label

## Checkpoints

Checkpoints were saved after each epoch:

```text
./checkpoints/latent_diffusion_epoch_1.pt
...
./checkpoints/latent_diffusion_epoch_50.pt

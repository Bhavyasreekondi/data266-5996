# DATA 266 — HW8

## Mini Diffusion Model: DDPM to DDIM

This assignment implements a simplified diffusion model on the MNIST dataset.
The model learns to predict Gaussian noise added during the forward diffusion
process and generates images using DDPM and DDIM reverse sampling.

## Student configuration

- Repository: `data266-5996`
- SID4: `5996`
- Seed: `5996`
- Assigned slice value: `996`
- HP_ID: `2`
- Focus class A: `6`
- Focus class B: `2`

The executed notebook used seed `5996`. The complete MNIST training set was
used, and slice-based or focus-class filtering was not applied.

## Configuration

- Dataset: MNIST
- Training samples: 60,000
- Image size: 28 × 28
- Image channels: 1
- Diffusion steps: 100
- Beta schedule: linear
- Beta start: 0.0001
- Beta end: 0.02
- Batch size: 128
- Epochs: 50
- Optimizer: Adam
- Learning rate: 0.001
- Loss function: Mean squared error
- Device: CUDA GPU
- DDPM sampling steps: 100
- DDIM sampling steps: 50 and 20
- DDIM eta: 0

## Model architecture

The timestep-conditioned CNN contains five convolutional layers:

- Conv2D: 1 → 64 channels
- Conv2D: 64 → 128 channels
- Conv2D: 128 → 128 channels
- Conv2D: 128 → 64 channels
- Conv2D: 64 → 1 channel

The model uses:

- ReLU activations
- SiLU activation in the timestep embedding network
- Sinusoidal timestep embeddings
- Timestep conditioning in intermediate convolutional layers

The verified model shapes were:

```text
Input shape:  torch.Size([4, 1, 28, 28])
Output shape: torch.Size([4, 1, 28, 28])

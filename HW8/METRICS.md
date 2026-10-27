
## `METRICS.md`

```markdown
# HW8 Metrics

## Assignment

Mini Diffusion Model: DDPM to DDIM on MNIST

## Experimental configuration

| Parameter | Value |
|---|---:|
| SID4 | 5996 |
| Seed | 5996 |
| Assigned slice value | 996 |
| HP_ID | 2 |
| Focus class A | 6 |
| Focus class B | 2 |
| Dataset | MNIST |
| Training samples | 60,000 |
| Diffusion steps, T | 100 |
| Configured beta start | 0.0001 |
| Configured beta end | 0.02 |
| Batch size | 128 |
| Epochs | 50 |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Loss | MSE |
| Device | CUDA |
| DDPM sampling steps | 100 |
| DDIM sampling steps | 50 and 20 |
| DDIM eta | 0 |

## Model architecture

| Component | Value |
|---|---|
| Input shape | `[4, 1, 28, 28]` |
| Output shape | `[4, 1, 28, 28]` |
| Convolutional layers | 5 |
| Activations | ReLU and SiLU |
| Timestep conditioning | Sinusoidal embedding |
| Output | Predicted noise tensor |

## Beta schedule values

| Quantity | Executed value |
|---|---:|
| First beta | 0.000099999997 |
| Last beta | 0.019999999553 |

## Training metrics

| Metric | Value |
|---|---:|
| Initial training loss | 0.162754 |
| Final training loss | 0.058772 |
| Best training loss | 0.058718 |
| Number of epochs | 50 |
| Batch size | 128 |
| Number of training samples | 60,000 |

## Sampling metrics

Sampling time was measured while generating 16 images.

| Method | Reverse steps | Sampling time |
|---|---:|---:|
| DDPM | 100 | 0.110 seconds |
| DDIM | 50 | 0.069 seconds |
| DDIM | 20 | 0.028 seconds |

## Relative speed

| Comparison | Result |
|---|---|
| DDIM 50 versus DDPM | 0.041 seconds faster |
| DDIM 20 versus DDPM | 0.082 seconds faster |
| DDIM 50 speed improvement | Approximately 37.3% |
| DDIM 20 speed improvement | Approximately 74.5% |
| Fastest method | DDIM with 20 steps |
| Slowest method | DDPM with 100 steps |

## Qualitative observations

- Forward diffusion progressively added noise at timesteps 0, 10, 25, 50, 75, and 99.
- The training loss decreased from 0.162754 to 0.058772.
- DDPM generated images using 100 reverse steps.
- DDIM generated images using 50 and 20 reverse steps.
- DDIM sampling was faster than DDPM.
- DDIM with 20 steps was the fastest configuration.
- Fewer DDIM steps may reduce image detail or make digits blurrier.
- With eta equal to 0, DDIM was deterministic for fixed initial noise.

## Required figures

The following figures should be stored in the `Figures/` directory:

1. `forward_diffusion.png`
2. `training_loss.png`
3. `ddpm_samples.png`
4. `ddim_50_samples.png`
5. `ddim_20_samples.png`
6. `sampling_comparison.png`

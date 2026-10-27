# AI Use Disclosure — HW8

## Assignment

Mini Diffusion Model: DDPM to DDIM on MNIST

## AI tool used

ChatGPT was used as a coding and study assistant.

## Assistance provided

AI assistance was used to:

1. Explain the DDPM forward diffusion equation.
2. Explain the DDPM reverse sampling equation.
3. Explain DDIM sampling and timestep skipping.
4. Structure the PyTorch notebook.
5. Design a timestep-conditioned CNN.
6. Explain the required convolutional architecture.
7. Help debug tensor shapes and sampling procedures.
8. Prepare the repository documentation files.
9. Explain the comparison between DDPM and DDIM.

## Prompts used

- Help me implement a simplified DDPM model on MNIST.
- Explain the forward diffusion equation step by step.
- Implement DDPM reverse sampling in PyTorch.
- Modify the DDPM sampler to implement DDIM sampling with 50 and 20 steps.
- Help me prepare the README, metrics, run log, and AI use documentation.

## Human verification

The notebook was executed in Google Colab using a CUDA GPU.

I verified that:

- The MNIST dataset loaded successfully with 60,000 training samples.
- The forward diffusion visualization worked correctly.
- The CNN contained five convolutional layers.
- The CNN used nonlinear activations.
- The CNN was conditioned on the timestep.
- The CNN output shape matched the input image shape.
- The training loss decreased from 0.162754 to 0.058772.
- The best recorded training loss was 0.058718.
- DDPM samples were generated using 100 reverse steps.
- DDIM samples were generated using 50 and 20 reverse steps.
- DDPM and DDIM sampling times were recorded.
- The model checkpoint was saved as `mini_ddpm_mnist.pth`.

## Final experimental values verified

- Seed: `5996`
- Device: `cuda`
- Dataset: MNIST
- Diffusion steps: `100`
- Batch size: `128`
- Epochs: `50`
- DDPM sampling time: `0.110 seconds`
- DDIM 50-step sampling time: `0.069 seconds`
- DDIM 20-step sampling time: `0.028 seconds`

AI-generated suggestions were treated as assistance only. The final code was
reviewed, executed, and verified by me.

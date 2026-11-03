# AI_USE.md

Generative AI assistance was used for HW9 to:

1. Organize the notebook into the required sections:
   - VAE compression
   - Latent diffusion
   - Conditional U-Net
   - Classifier-free guidance
   - Training
   - Sampling
   - Analysis

2. Provide implementation assistance for:
   - Loading the pretrained Stable Diffusion VAE
   - Encoding and decoding FashionMNIST images
   - Scaling latent representations
   - Implementing the forward diffusion process
   - Creating sinusoidal timestep embeddings
   - Implementing residual blocks and self-attention
   - Adding class conditioning and null-label dropout
   - Implementing classifier-free guidance sampling
   - Creating visualizations and checkpoint-saving code

3. Help debug and explain technical issues related to:
   - Converting grayscale FashionMNIST images to three channels
   - Resizing images to 128 x 128
   - Using the VAE latent scaling factor
   - Managing GPU memory through cached float16 latents
   - Saving model checkpoints after every epoch

The final notebook was executed and the outputs, training loss values, reconstruction MSE, model architecture, and generated sample grids were verified.

The student is responsible for understanding, explaining, and defending the implementation and results.

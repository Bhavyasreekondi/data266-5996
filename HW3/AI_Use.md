# AI Use Appendix - HW3

### 1. Which parts did you use an assistant for, and which did you write yourself?

I used AI assistance mainly to understand the assignment requirements, plan the implementation step by step, debug errors that occurred while running the notebook, and understand how to structure the prompt-engineering and attention experiments. I also used AI to suggest alternative approaches when the initial code did not work and to help organize the documentation and findings. I executed all the code myself, checked the outputs, compared the prompt results, generated the attention heatmaps, and verified the final implementation before including it in the submission.

### 2. Give one specific thing it produced that was wrong.

One incorrect approach suggested while implementing causal masking assumed that an attention-score variable called `scores` was already available:

    masked_scores = scores.masked_fill(
        causal_mask == 0,
        float("-inf")
    )

When I executed this code, I received:

    NameError: name 'scores' is not defined

### 3. How did you find out? What did the failure look like?

I found the problem by executing the code in the notebook. The causal-masking cell stopped with a `NameError`, so the masked attention weights and heatmap could not be generated. I checked the traceback and the previous cells and realized that `scores` had not been created in that scope. This showed that the suggested code could not simply be added independently and needed to be integrated with the actual attention implementation.

### 4. What did you change, and why does your version work?

Instead of directly using the undefined `scores` variable, I used an approach where the attention scores are first calculated from the learned query and key projections using scaled dot-product attention. I then created the lower-triangular causal mask and applied it to those scores before softmax. This ensures that future token positions are masked before the attention probabilities are calculated. After making the change, the code executed successfully and I verified the result using the causal-attention heatmap, where tokens were prevented from attending to future positions.

Overall, I used AI as an assistant for explanations, debugging, code suggestions, and alternative approaches, but I tested the suggestions in the notebook and modified them when they did not match my implementation or produced incorrect results.

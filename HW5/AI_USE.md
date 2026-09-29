# AI Use Disclosure

## Tool Used

ChatGPT (OpenAI)

## How AI Was Used

AI assistance was used during this assignment for:

- Understanding the steps required to fine-tune FLAN-T5-small using LoRA.
- Debugging PEFT/LoRA and Hugging Face Trainer configuration issues.
- Diagnosing invalid training runs that produced zero training loss and NaN validation loss.
- Reviewing LoRA configurations for ranks r=4 and r=16.
- Organizing inference code so that the same test dialogues were used for model comparison.
- Interpreting the observed training and validation losses.
- Organizing the comparison between the pretrained baseline and LoRA fine-tuned models.
- Preparing the repository documentation and artifact structure.

## Verification

All code was executed in Google Colab and the resulting outputs were reviewed. Invalid training runs were not used for the reported final metrics. The final reported results are based on successful executed runs preserved in the submitted notebook and documented in RUN_LOG.txt and METRICS.md.

## Modifications and Responsibility

AI-generated suggestions were reviewed and adapted during implementation. The submitted notebook contains the code that was actually executed, and the reported results correspond to the observed experiment outputs.

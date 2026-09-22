HW4 — METRICS

Mini GPT Configuration

Metric

Value

Vocabulary size

49

Sequence length

128

Hidden dimension

128

Attention heads

4

Decoder layers

2

Feedforward dimension

512

Total parameters

425,777

Trainable parameters

425,777

Training Configuration

Metric

Value

Optimizer

Adam

Learning rate

0.0003

Batch size

64

Epochs

5

Loss function

CrossEntropyLoss

Training Results

Epoch

Average training loss

1

3.3937

2

2.8129

3

2.5608

4

2.3909

5

2.2806

Initial loss: 3.3937

Final loss: 2.2806

Loss reduction: 1.1131

Sampling Results

Prompt: ROMEO:

Greedy decoding: deterministic but highly repetitive.

Temperature sampling tested at 0.5, 1.0, and 1.5.

Top-k sampling tested at k=5 and k=20.

Among the tested outputs, temperature 0.5 produced the most coherent result.

Temperature 1.5 produced the most diverse result.

Increasing temperature increased diversity but reduced textual consistency.

Increasing k from 5 to 20 increased the set of possible next characters and produced more varied output.

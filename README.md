# data266-5996

# DATA 266 - Homework Repository

This repository contains coursework and experimental results for **DATA 266**.  
The repository is organized by homework assignment, with each assignment containing the required source code, executed notebooks, measurements, run logs, reports, and AI-use documentation.

## Student Parameters

The course requires each student to derive experimental parameters from the last four digits of their student ID.

For this repository:

| Parameter | Value |
|---|---:|
| SID4 | 5996 |
| SEED | 5996 |
| SLICE | 996 |
| HP_ID | 2 |
| CLS_A | 6 |
| CLS_B | 2 |

These parameters are used to ensure that the experiments are reproducible and student-specific.

---

# Homework 1

HW1 explores two main topics:

1. **Neural-network implementation and experimentation using PyTorch and TensorFlow**
2. **GPU parallel computing using CUDA matrix multiplication**

The assignment focuses on reproducibility, controlled experimentation, performance measurement, profiling, and interpretation of results.

---

## Part 1 - Autoregressive Models

The assignment begins with a conceptual discussion of **autoregressive models**.

Autoregressive models predict the next element of a sequence using previously observed elements. They are commonly used in applications such as:

- Language modeling
- Time-series forecasting
- Sequential data generation
- Demand and sales forecasting

---

## Part 2 - Neural Network Experiments

A binary classification dataset containing **759 samples and 8 input features** is used for the neural-network experiments.

The dataset is divided using a fixed:

- **70% training set**
- **15% validation set**
- **15% test set**

The split uses:

`random_state = 5996`

The same data split is maintained across all model comparisons so that differences in performance are caused by the experimental configuration rather than different train/test samples.

### Baseline Model

The required baseline neural network uses:

- Hidden layers: `[64, 32]`
- Learning rate: `0.001`
- Epochs: `30`

The baseline architecture is implemented independently using:

- **PyTorch**
- **TensorFlow**

### Assigned Hyperparameter Experiment

The assigned hyperparameter configuration is determined by:

`HP_ID = 2`

For HP_ID 2, the modified model uses:

- Hidden layers: `[64, 32]`
- Learning rate: `0.003`
- Epochs: `30`

Therefore, the primary experimental variable for this assignment is the change in learning rate from:

`0.001 → 0.003`

while keeping the neural-network architecture and number of training epochs unchanged.

### Repeated Training

To evaluate variability caused by model initialization and training, each baseline and modified model is trained using three training seeds:

- `5996`
- `5997`
- `5998`

The dataset split remains fixed while only the training seed changes.

Mean and standard deviation of test accuracy are reported for each model and framework.

### Neural Network Results

| Framework | Model | Learning Rate | Mean Test Accuracy | Std. Test Accuracy |
|---|---|---:|---:|---:|
| PyTorch | Baseline | 0.001 | 0.649123 | 0.000000 |
| PyTorch | HP_ID 2 Modified | 0.003 | 0.751462 | 0.004135 |
| TensorFlow | Baseline | 0.001 | 0.733918 | 0.008270 |
| TensorFlow | HP_ID 2 Modified | 0.003 | 0.728070 | 0.007162 |

The results demonstrate that the effect of increasing the learning rate was framework-dependent. The modified learning rate substantially improved the PyTorch model's mean test accuracy, while the TensorFlow model showed a small decrease in mean test accuracy.

Training and validation loss curves are also included in the executed notebook to compare the baseline and modified configurations and examine possible overfitting or underfitting behavior.

---

## Part 3 - CUDA Matrix Multiplication

The second experimental component implements matrix multiplication using both:

- A CPU implementation
- A CUDA GPU implementation

The purpose is to compare CPU execution with GPU parallel execution while separately measuring computation and memory-transfer costs.

### CUDA Kernel Design

The CUDA implementation uses:

- `16 × 16` thread blocks
- `256` threads per block
- A two-dimensional CUDA grid
- Shared-memory tiling
- One output matrix element computed by each thread

The grid dimensions are determined from the matrix size so that all elements of the output matrix are covered.

### Matrix Sizes

Performance is evaluated using:

- `N = 256`
- `N = 1024`
- `N = 4096`

For each matrix size, the program measures:

- CPU execution time
- GPU kernel execution time
- Host-to-device (H2D) transfer time
- Device-to-host (D2H) transfer time
- GPU end-to-end execution time
- CPU-to-GPU speedup
- Maximum numerical error

Each benchmark size is measured **three times**, and the arithmetic mean is used for the final performance results.

### CUDA Performance Results

| Matrix Size | Mean CPU Time (ms) | Mean GPU Kernel Time (ms) | Mean H2D+D2H (ms) | Mean GPU End-to-End (ms) | Speedup |
|---:|---:|---:|---:|---:|---:|
| 256 | 3.0701 | 0.0541 | 0.2788 | 0.3330 | 9.22x |
| 1024 | 199.7894 | 2.6487 | 2.9977 | 5.6464 | 35.39x |
| 4096 | 20256.7097 | 199.5739 | 47.0412 | 246.6151 | 82.14x |

The GPU was already faster than the CPU at the smallest tested matrix size. The performance advantage increased substantially as matrix size increased because larger workloads provide more parallel computation and better amortize host-device transfer overhead.

---

## CUDA Correctness Verification

The GPU output is compared against the CPU output to verify correctness.

| Matrix Size | Maximum Absolute Error |
|---:|---:|
| 256 | 0.0000 |
| 1024 | 0.0001 |
| 4096 | 0.0004 |

The small differences are consistent with floating-point rounding caused by different arithmetic execution orders on the CPU and GPU.

---

## GPU Profiling

The CUDA implementation was profiled on an **NVIDIA Tesla T4** using **NVIDIA Nsight Compute (`ncu`)**.

Profiling was used to examine characteristics such as:

- Kernel launch configuration
- Block and grid sizes
- GPU occupancy
- Compute utilization
- Memory behavior

Profiler-instrumented execution times are kept separate from the primary benchmark measurements because profiling introduces instrumentation and kernel replay overhead.

The final timing results therefore come from normal repeated executions of the CUDA program rather than execution under Nsight Compute.

---

# Reproducibility

Reproducibility is maintained throughout HW1 by:

- Using `SEED = 5996`
- Keeping the 70/15/15 dataset split fixed
- Using the same split across model comparisons
- Training with seeds `5996`, `5997`, and `5998`
- Reporting aggregate neural-network performance
- Warming up the GPU before benchmark measurements
- Repeating CUDA measurements three times
- Recording raw console output in `RUN_LOG.txt`
- Recording final measurements in `METRICS.md`

---

# Repository Structure

The HW1 portion of the repository is organized as follows:

```text
data266-5996/
│
├── README.md
│
└── HW1/
    ├── neural_networks.ipynb
    ├── cuda.ipynb
    ├── matrix_mul.cu
    ├── RUN_LOG.txt
    ├── METRICS.md
    ├── AI_USE.md
    └── HW1_Report.pdf
```

### File Descriptions

**`neural_networks.ipynb`**  
Executed notebook containing dataset preparation, PyTorch and TensorFlow implementations, baseline and modified experiments, repeated-seed evaluation, loss curves, and model comparisons.

**`cuda.ipynb`**  
Executed notebook containing CUDA environment information, compilation, benchmark execution, repeated timing measurements, correctness verification, profiling output, and performance analysis.

**`matrix_mul.cu`**  
CUDA/C++ source code implementing CPU and GPU matrix multiplication.

**`RUN_LOG.txt`**  
Raw console output from the runs used to produce the reported measurements, along with relevant environment information and errors encountered during experimentation.

**`METRICS.md`**  
Summary of the neural-network and CUDA measurements reported for HW1.

**`AI_USE.md`**  
Documentation of AI-assistant usage, verification, corrections, and changes made during the assignment.

**`HW1_Report.pdf`**  
Final written report containing the experimental methodology, results, figures, analysis, and conclusions.

---

# Assignment Versioning

Course assignments are versioned using Git tags.

The final HW1 submission is identified by:

`hw1`

Incremental commits are used throughout development to preserve the progression of implementation, experimentation, measurement, and documentation.

---

## Tools and Technologies

HW1 uses:

- Python
- NumPy
- PyTorch
- TensorFlow
- CUDA C/C++
- NVIDIA CUDA Toolkit
- NVIDIA Nsight Compute
- Google Colab
- Git
- GitHub

# DATA 266 - HW1 Metrics

## Personal Parameters

| Parameter | Value |
|---|---:|
| SID4 | 5996 |
| SEED | 5996 |
| SLICE | 996 |
| HP_ID | 2 |
| CLS_A | 6 |
| CLS_B | 2 |

---

## 1. Dataset Summary

The neural-network experiments used a binary classification dataset with 759 samples, 8 input features, and one binary target variable.

| Metric | Value |
|---|---:|
| Total samples | 759 |
| Input features | 8 |
| Class 0 samples | 263 |
| Class 1 samples | 496 |
| Training split | 70% |
| Validation split | 15% |
| Test split | 15% |
| Split random state | 5996 |

The same fixed data split was used for all baseline and modified model comparisons.

---

## 2. Neural Network Configurations

### Baseline Configuration

| Parameter | Value |
|---|---|
| Hidden layers | [64, 32] |
| Learning rate | 0.001 |
| Epochs | 30 |

### HP_ID 2 Modified Configuration

For `HP_ID = 2`, the assigned modification changes the learning rate while keeping the architecture and number of epochs unchanged.

| Parameter | Baseline | Modified |
|---|---:|---:|
| Hidden layers | [64, 32] | [64, 32] |
| Learning rate | 0.001 | 0.003 |
| Epochs | 30 | 30 |

### Training Seeds

Each configuration was trained using:

- 5996
- 5997
- 5998

The dataset split remained fixed while the training seed changed.

---

## 3. Neural Network Results

| Framework | Model | Hidden Layers | Learning Rate | Epochs | Mean Test Accuracy | Std. Test Accuracy |
|---|---|---|---:|---:|---:|---:|
| PyTorch | Baseline | [64, 32] | 0.001 | 30 | 0.649123 | 0.000000 |
| PyTorch | HP_ID 2 Modified | [64, 32] | 0.003 | 30 | 0.751462 | 0.004135 |
| TensorFlow | Baseline | [64, 32] | 0.001 | 30 | 0.733918 | 0.008270 |
| TensorFlow | HP_ID 2 Modified | [64, 32] | 0.003 | 30 | 0.728070 | 0.007162 |

### Neural Network Result Summary

For PyTorch, increasing the learning rate from 0.001 to 0.003 increased mean test accuracy from 0.649123 to 0.751462.

For TensorFlow, increasing the learning rate from 0.001 to 0.003 slightly decreased mean test accuracy from 0.733918 to 0.728070.

The results indicate that the effect of the assigned learning-rate modification was framework-dependent.

---

## 4. CUDA Benchmark Configuration

| Parameter | Value |
|---|---|
| GPU | NVIDIA Tesla T4 |
| Matrix sizes | 256, 1024, 4096 |
| Thread block | 16 x 16 |
| Threads per block | 256 |
| Benchmark repetitions | 3 |
| Aggregation | Arithmetic mean |
| Profiler | NVIDIA Nsight Compute (`ncu`) |

GPU warm-up was performed before the measured benchmark execution.

The final timing measurements were obtained from normal execution of `./matrix_mul` without profiler instrumentation.

---

## 5. CUDA Timing Results

Each matrix size was measured three times. The table reports the arithmetic mean of the three normal benchmark runs.

| Matrix Size (N) | Mean CPU Time (ms) | Mean GPU Kernel Time (ms) | Mean H2D + D2H Time (ms) | Mean GPU End-to-End Time (ms) | End-to-End Speedup |
|---:|---:|---:|---:|---:|---:|
| 256 | 3.0701 | 0.0541 | 0.2788 | 0.3330 | 9.22x |
| 1024 | 199.7894 | 2.6487 | 2.9977 | 5.6464 | 35.39x |
| 4096 | 20256.7097 | 199.5739 | 47.0412 | 246.6151 | 82.14x |

GPU end-to-end time is calculated as:

`GPU Kernel Time + H2D Time + D2H Time`

End-to-end speedup is calculated as:

`Mean CPU Time / Mean GPU End-to-End Time`

---

## 6. CUDA Correctness

The CUDA output was compared against the CPU matrix-multiplication output.

| Matrix Size (N) | Maximum Absolute Error |
|---:|---:|
| 256 | 0.0000 |
| 1024 | 0.0001 |
| 4096 | 0.0004 |

The small maximum absolute errors indicate that the CUDA implementation produced results numerically consistent with the CPU implementation. Small differences are expected because floating-point arithmetic may be performed in a different order on the CPU and GPU.

---

## 7. CPU-GPU Performance Comparison

| Matrix Size (N) | End-to-End Speedup |
|---:|---:|
| 256 | 9.22x |
| 1024 | 35.39x |
| 4096 | 82.14x |

The GPU was already faster than the CPU at the smallest tested matrix size, `N = 256`. The performance advantage increased as matrix size increased because the larger workloads provided more parallel computation and reduced the relative impact of host-device transfer overhead.

The exact CPU-GPU crossover point cannot be determined from these measurements because matrix sizes smaller than 256 were not tested.

---

## 8. CUDA Profiling Summary

NVIDIA Nsight Compute (`ncu`) was used to profile the CUDA matrix-multiplication kernel.

The profiler confirmed:

- 16 x 16 thread blocks
- 256 threads per block
- High achieved occupancy for the largest tested workload
- Substantial GPU utilization during matrix multiplication

Profiler-instrumented execution times were not used in the final CUDA timing table because Nsight Compute instruments and replays kernels while collecting hardware metrics.

The final performance measurements therefore come from the three normal `./matrix_mul` benchmark runs.

---

## 9. Final Metrics Summary

### Neural Networks

- Best PyTorch mean test accuracy: **0.751462**
- PyTorch configuration: **HP_ID 2 Modified, LR = 0.003**
- Best TensorFlow mean test accuracy: **0.733918**
- TensorFlow configuration: **Baseline, LR = 0.001**

### CUDA

- N = 256 speedup: **9.22x**
- N = 1024 speedup: **35.39x**
- N = 4096 speedup: **82.14x**
- Largest measured maximum absolute error: **0.0004**

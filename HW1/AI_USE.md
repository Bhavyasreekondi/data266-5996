# DATA 266 - HW1 AI Use Disclosure

## AI Assistant Used

I used **ChatGPT** as an AI assistant while completing HW1.

I used ChatGPT primarily to:

- Clarify assignment requirements and organize the required deliverables.
- Understand the implementation of the neural networks in PyTorch and TensorFlow.
- Review the experimental setup for the baseline and HP_ID-modified models.
- Help interpret training/validation loss curves and test accuracy results.
- Understand CUDA thread blocks, grids, shared-memory tiling, and GPU timing.
- Help interpret NVIDIA Nsight Compute profiling output.
- Organize the final measurement tables and written explanations.

I reviewed the generated suggestions and compared them with my actual program outputs before including results in the assignment.

---

## Specific AI Error Identified

One specific issue occurred during the CUDA performance analysis.

After I ran the CUDA program using NVIDIA Nsight Compute with:

```bash
ncu --target-processes all ./matrix_mul
```

ChatGPT initially treated the execution times shown during the profiling run as if they were normal CUDA benchmark timings.

For example, the profiler execution showed GPU kernel times that were much larger than the timings obtained when the CUDA program was executed normally.

This interpretation was misleading because NVIDIA Nsight Compute instruments and replays CUDA kernels while collecting profiling metrics. Therefore, execution under `ncu` contains profiling overhead and should not be treated as the normal runtime of the CUDA program.

---

## How I Identified the Problem

I identified the issue by running the CUDA executable normally without Nsight Compute:

```bash
./matrix_mul
```

The normal execution produced substantially smaller GPU kernel times than the execution performed under `ncu`.

For example, during normal execution the GPU kernel time for `N = 4096` was approximately 200 ms, while the profiler execution reported a much larger value.

The Nsight Compute output also showed that the kernel was processed through multiple profiling passes. This confirmed that the profiler was adding instrumentation and replay overhead.

I therefore determined that the profiler timings should not be used as the primary CPU-versus-GPU benchmark measurements.

---

## What I Changed

I separated CUDA **benchmarking** from CUDA **profiling**.

For performance benchmarking, I executed:

```bash
./matrix_mul
```

without profiler instrumentation.

I then repeated the normal benchmark three times for each required matrix size:

- `N = 256`
- `N = 1024`
- `N = 4096`

I reported the arithmetic mean of the three normal runs for:

- CPU execution time
- GPU kernel time
- H2D + D2H transfer time
- GPU end-to-end time
- End-to-end speedup

I retained the NVIDIA Nsight Compute output only for profiling information such as kernel configuration, occupancy, and GPU utilization.

This correction ensured that the final performance table represents normal program execution rather than profiler-instrumented execution.

---

## Verification of AI-Generated Assistance

I verified the important numerical results against the actual outputs produced by my notebooks and CUDA program.

For the neural-network experiments, I checked the reported mean and standard deviation of test accuracy against the results from the three training seeds.

For the CUDA experiment, I checked that:

```text
GPU End-to-End Time =
GPU Kernel Time + H2D Time + D2H Time
```

and calculated speedup using:

```text
Speedup =
CPU Time / GPU End-to-End Time
```

I also compared the GPU matrix output against the CPU matrix output using the maximum absolute error to verify numerical correctness.

The final submitted measurements are based on the actual experimental outputs rather than unverified AI-generated values.

---

## Summary

ChatGPT was used as a supporting tool for understanding concepts, reviewing experimental methodology, interpreting results, and organizing documentation. I independently executed the code, reviewed the outputs, identified the CUDA profiling-versus-benchmarking issue, and corrected the methodology before reporting the final results.

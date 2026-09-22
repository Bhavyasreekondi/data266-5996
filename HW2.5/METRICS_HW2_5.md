# METRICS.md

# HW2.5 — GPU Assignment I: Precision, Bandwidth, and the Cost of Attention

## Hardware Note

The assignment specifies an NVIDIA RTX 5090 or RTX 4090 workstation. The
measurements in this submission were collected on NVIDIA A100-SXM4-40GB GPUs
provided by Google Colab. Results are reported using the actual GPU UUIDs and
must not be interpreted as RTX 4090/5090 measurements.

## Table HW2.5.1 — Summary

| Measurement | Your GPU | Notes |
|---|---:|---|
| Peak achieved TFLOPS (BF16) | **273.60 TFLOPS** | N=8192; UUID `GPU-f5e42ed3-9b0a-7003-62dd-4fb887642ee2` |
| % of theoretical peak (BF16) | **87.69%** | 273.596 / 312 TFLOPS; UUID `GPU-f5e42ed3-9b0a-7003-62dd-4fb887642ee2` |
| Effective bandwidth | **1381.25 GB/s** | 88.83% of 1555 GB/s; UUID `GPU-f5e42ed3-9b0a-7003-62dd-4fb887642ee2` |
| Naive attention OOM length | **25,344 succeeds; 25,600 fails** | OOM bracket, not an exact single-token boundary; UUID `GPU-4a3d609e-89cf-640b-8162-5adf66e41fde` |
| Fused attention OOM length | **No OOM observed through 4,194,304 tokens** | 4,194,304 is the largest tested successful length, not an OOM point; UUID `GPU-4a3d609e-89cf-640b-8162-5adf66e41fde` |
| Steady-state / peak throughput | **98.49%** | 257.66 TFLOPS final 5 min / 261.61 TFLOPS first 30 s; UUID `GPU-f5e42ed3-9b0a-7003-62dd-4fb887642ee2` |
| Throttle onset | **None observed** | No clear thermal-throttling evidence in collected telemetry; UUID `GPU-f5e42ed3-9b0a-7003-62dd-4fb887642ee2` |

## Part B — Precision and Achieved Throughput

Dense matrix multiplication was benchmarked at N = 1024, 4096, 8192, and
16384 in FP32, TF32, FP16, and BF16 using 5 warm-up runs and 20 timed
repetitions.

The highest measured BF16 throughput was **273.596 TFLOPS at N=8192**,
corresponding to **87.691%** of the 312 TFLOPS theoretical BF16 value used
for this experiment.

Peak observed throughput by precision:

| Precision | Peak achieved TFLOPS | Matrix size |
|---|---:|---:|
| FP32 | 19.146 | 8192 |
| TF32 | 131.194 | 4096 |
| FP16 | 265.092 | 8192 |
| BF16 | 273.596 | 8192 |

Small matrices did not reach peak throughput because fixed launch overhead and
insufficient parallel work have a larger relative impact. Throughput increased
with matrix size and then approached a plateau.

### Lower-precision attempt

PyTorch exposed the `Float8_e4m3fn` dtype, but the attempted FP8 matrix
multiplication failed with:

`NotImplementedError: "addmm_cuda" not implemented for 'Float8_e4m3fn'`

## Part C — Bandwidth vs. Compute

The memory-bound operation was FP32 elementwise addition over 200,000,000
elements. It achieved **1381.25 GB/s**, or **88.83%** of the 1555 GB/s
reference bandwidth.

| Operation | Arithmetic intensity | Roofline classification |
|---|---:|---|
| FP32 elementwise addition | 0.0833 FLOPs/byte | Memory-bound |
| BF16 N=16384 matrix multiplication | 5461.33 FLOPs/byte | Compute-bound |

The BF16 roofline ridge point used in the analysis was **200.64
FLOPs/byte**. The elementwise operation falls well below the ridge point,
while the matrix multiplication falls well above it.

## Part D — Cost of Attention

Naive scaled dot-product attention used batch size 1, 4 heads, head dimension
64, and FP16. The required coarse sequence-length sweep included 512, 1024,
2048, 4096, 8192, and 16384 tokens.

The refined naive-attention OOM boundary was bracketed between **25,344
tokens (success)** and **25,600 tokens (failure)**.

The fitted quadratic coefficient for naive peak-memory growth was approximately
**5.9613e-08 GB/token^2**, with **R^2 approximately 1.0**, confirming the
quadratic memory-growth term from the measured data.

For fused/memory-efficient scaled dot-product attention, no OOM was observed
through **4,194,304 tokens**. That sequence length succeeded and therefore is
not reported as an exact fused-attention OOM boundary.

The per-length naive/fused latency and speedup measurements are preserved in
`part_d_attention_comparison.csv`.

## Part E — Sustained Load and Thermal Behavior

The sustained-load experiment ran for approximately **20.07 minutes**, with
telemetry sampled every **5 seconds**.

| Thermal/throughput metric | Result |
|---|---:|
| Telemetry samples | 240 |
| Peak throughput, first 30 s | 261.61 TFLOPS |
| Steady-state throughput, final 5 min | 257.66 TFLOPS |
| Steady-state / peak | 98.49% |
| Maximum temperature | 72 C |
| Average loaded temperature | 71.2 C |
| Maximum power | 411.34 W |
| Average loaded power | 397.33 W |
| Loaded SM clock range | 1230–1260 MHz |
| Throttle onset | None observed |

No clear evidence of thermal throttling was observed in the collected
telemetry. Throughput and loaded SM clock remained relatively stable after
thermal steady state was reached.

## UUID Traceability

Parts B, C, and E:
`GPU-f5e42ed3-9b0a-7003-62dd-4fb887642ee2`

Final Part D boundary artifacts:
`GPU-4a3d609e-89cf-640b-8162-5adf66e41fde`

Detailed console results and relevant failures are recorded in `RUN_LOG.txt`.

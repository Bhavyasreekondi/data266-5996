
#include <cuda_runtime.h>
#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>
#include <cmath>

#define TILE 16

__global__ void matrixMulKernel(
    const float* A,
    const float* B,
    float* C,
    int N
) {
    __shared__ float tileA[TILE][TILE];
    __shared__ float tileB[TILE][TILE];

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float sum = 0.0f;

    int numberOfTiles = (N + TILE - 1) / TILE;

    for (int t = 0; t < numberOfTiles; ++t) {

        int A_col = t * TILE + threadIdx.x;
        int B_row = t * TILE + threadIdx.y;

        if (row < N && A_col < N)
            tileA[threadIdx.y][threadIdx.x] =
                A[row * N + A_col];
        else
            tileA[threadIdx.y][threadIdx.x] = 0.0f;

        if (B_row < N && col < N)
            tileB[threadIdx.y][threadIdx.x] =
                B[B_row * N + col];
        else
            tileB[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE; ++k) {
            sum +=
                tileA[threadIdx.y][k] *
                tileB[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < N && col < N)
        C[row * N + col] = sum;
}


void cpuMatrixMultiply(
    const float* A,
    const float* B,
    float* C,
    int N
) {
    for (int i = 0; i < N; ++i) {
        for (int k = 0; k < N; ++k) {

            float value = A[i * N + k];

            for (int j = 0; j < N; ++j) {
                C[i * N + j] +=
                    value * B[k * N + j];
            }
        }
    }
}


void runBenchmark(int N) {

    size_t elements =
        static_cast<size_t>(N) * N;

    size_t bytes =
        elements * sizeof(float);

    std::vector<float> A(elements);
    std::vector<float> B(elements);

    std::vector<float> C_cpu(
        elements,
        0.0f
    );

    std::vector<float> C_gpu(
        elements,
        0.0f
    );

    srand(5996);

    for (size_t i = 0; i < elements; ++i) {

        A[i] =
            static_cast<float>(rand()) /
            RAND_MAX;

        B[i] =
            static_cast<float>(rand()) /
            RAND_MAX;
    }

    // CPU timing
    auto cpuStart =
        std::chrono::high_resolution_clock::now();

    cpuMatrixMultiply(
        A.data(),
        B.data(),
        C_cpu.data(),
        N
    );

    auto cpuEnd =
        std::chrono::high_resolution_clock::now();

    double cpuMs =
        std::chrono::duration<double, std::milli>(
            cpuEnd - cpuStart
        ).count();


    // GPU memory
    float* d_A;
    float* d_B;
    float* d_C;

    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);


    cudaEvent_t start;
    cudaEvent_t stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);


    // Host to Device
    cudaEventRecord(start);

    cudaMemcpy(
        d_A,
        A.data(),
        bytes,
        cudaMemcpyHostToDevice
    );

    cudaMemcpy(
        d_B,
        B.data(),
        bytes,
        cudaMemcpyHostToDevice
    );

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float h2dMs;

    cudaEventElapsedTime(
        &h2dMs,
        start,
        stop
    );


    dim3 threadsPerBlock(
        TILE,
        TILE
    );

    dim3 blocksPerGrid(
        (N + TILE - 1) / TILE,
        (N + TILE - 1) / TILE
    );


    // Warm up
    matrixMulKernel<<<
        blocksPerGrid,
        threadsPerBlock
    >>>(
        d_A,
        d_B,
        d_C,
        N
    );

    cudaDeviceSynchronize();


    // Kernel timing
    cudaEventRecord(start);

    matrixMulKernel<<<
        blocksPerGrid,
        threadsPerBlock
    >>>(
        d_A,
        d_B,
        d_C,
        N
    );

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float kernelMs;

    cudaEventElapsedTime(
        &kernelMs,
        start,
        stop
    );


    // Device to Host
    cudaEventRecord(start);

    cudaMemcpy(
        C_gpu.data(),
        d_C,
        bytes,
        cudaMemcpyDeviceToHost
    );

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float d2hMs;

    cudaEventElapsedTime(
        &d2hMs,
        start,
        stop
    );


    float transferMs =
        h2dMs + d2hMs;

    float gpuEndToEndMs =
        kernelMs + transferMs;

    double speedup =
        cpuMs / gpuEndToEndMs;


    // Correctness check
    float maxError = 0.0f;

    for (size_t i = 0; i < elements; ++i) {

        float error =
            std::fabs(
                C_cpu[i] - C_gpu[i]
            );

        if (error > maxError)
            maxError = error;
    }


    std::cout
        << std::fixed
        << std::setprecision(4);

    std::cout << "\nMatrix size: "
              << N << " x " << N
              << std::endl;

    std::cout << "CPU time (ms): "
              << cpuMs
              << std::endl;

    std::cout << "GPU kernel time (ms): "
              << kernelMs
              << std::endl;

    std::cout << "H2D time (ms): "
              << h2dMs
              << std::endl;

    std::cout << "D2H time (ms): "
              << d2hMs
              << std::endl;

    std::cout << "H2D+D2H (ms): "
              << transferMs
              << std::endl;

    std::cout << "GPU end-to-end (ms): "
              << gpuEndToEndMs
              << std::endl;

    std::cout << "End-to-end speedup: "
              << speedup
              << "x"
              << std::endl;

    std::cout << "Maximum error: "
              << maxError
              << std::endl;


    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}


int main() {

    runBenchmark(256);
    runBenchmark(1024);
    runBenchmark(4096);

    return 0;
}

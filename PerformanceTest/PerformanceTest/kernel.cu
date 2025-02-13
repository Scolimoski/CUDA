
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include <iostream>
#include "cmath"

#include <cstdlib>
#include <ctime>
#include <chrono>

// Kernel CUDA para calcular a média
__global__ void calculateMean(const float* data, float* result, int N) {
    extern __shared__ float sharedData[]; // Memória compartilhada

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Carregar dados na memória compartilhada
    if (i < N) {
        sharedData[tid] = data[i];
    }
    else {
        sharedData[tid] = 0.0f; // Preencher com zero se fora do intervalo
    }
    __syncthreads(); // Sincronizar threads dentro do bloco

    // Redução na memória compartilhada
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sharedData[tid] += sharedData[tid + s];
        }
        __syncthreads(); // Sincronizar threads após cada etapa da redução
    }

    // Escrever o resultado do bloco
    if (tid == 0) {
        result[blockIdx.x] = sharedData[0];
    }
}

// Função para calcular a média na CPU
float calculateMeanCPU(const float* data, int N) {
    float sum = 0.0f;
    for (int i = 0; i < N; i++) {
        sum += data[i];
    }
    return sum / N;
}

int main() {
    const int N = 2000000; // Tamanho do vetor (aumentado para melhor comparação)
    const int blockSize = 256; // Tamanho do bloco
    const int gridSize = (N + blockSize - 1) / blockSize; // Tamanho da grade

    // Alocar memória no host
    float* h_random_signal = new float[N];
    float* h_result = new float[gridSize];

    // Gerar sinal aleatório
    std::srand(std::time(0)); // Inicializar a semente do gerador de números aleatórios
    for (int i = 0; i < N; i++) {
        h_random_signal[i] = static_cast<float>(std::rand()) / RAND_MAX; // Números aleatórios entre 0 e 1
    }

    // Calcular a média na CPU e medir o tempo
    auto start_cpu = std::chrono::high_resolution_clock::now();
    float cpu_mean = calculateMeanCPU(h_random_signal, N);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> cpu_time = end_cpu - start_cpu;

    // Alocar memória na GPU
    float* d_random_signal, * d_result;
    cudaMalloc((void**)&d_random_signal, N * sizeof(float));
    cudaMalloc((void**)&d_result, gridSize * sizeof(float));

    // Copiar dados para a GPU
    auto start_memcpy = std::chrono::high_resolution_clock::now();
    cudaMemcpy(d_random_signal, h_random_signal, N * sizeof(float), cudaMemcpyHostToDevice);
    auto end_memcpy = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> memcpy_time = end_memcpy - start_memcpy;

    // Executar o kernel e medir o tempo
    auto start_kernel = std::chrono::high_resolution_clock::now();
    calculateMean << <gridSize, blockSize, blockSize * sizeof(float) >> > (d_random_signal, d_result, N);
    cudaDeviceSynchronize(); // Sincronizar para garantir que o kernel terminou
    auto end_kernel = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> kernel_time = end_kernel - start_kernel;

    // Copiar resultado de volta para o host e medir o tempo
    auto start_memcpy_back = std::chrono::high_resolution_clock::now();
    cudaMemcpy(h_result, d_result, gridSize * sizeof(float), cudaMemcpyDeviceToHost);
    auto end_memcpy_back = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> memcpy_back_time = end_memcpy_back - start_memcpy_back;

    // Calcular a média final no host
    float gpu_mean = 0.0f;
    for (int i = 0; i < gridSize; i++) {
        gpu_mean += h_result[i];
    }
    gpu_mean /= N;

    // Exibir os resultados e tempos
    std::cout << "Media na CPU: " << cpu_mean << std::endl;
    std::cout << "Tempo na CPU: " << cpu_time.count() << " segundos" << std::endl;
    std::cout << "Media na GPU: " << gpu_mean << std::endl;
    std::cout << "Tempo de copia para a GPU: " << memcpy_time.count() << " segundos" << std::endl;
    std::cout << "Tempo de execução do kernel: " << kernel_time.count() << " segundos" << std::endl;
    std::cout << "Tempo de copia de volta para a CPU: " << memcpy_back_time.count() << " segundos" << std::endl;
    std::cout << "Tempo total na GPU: " << memcpy_time.count() + kernel_time.count() + memcpy_back_time.count() << " segundos" << std::endl;

    // Liberar memória
    delete[] h_random_signal;
    delete[] h_result;
    cudaFree(d_random_signal);
    cudaFree(d_result);

    return 0;
}
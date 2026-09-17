#include "radix.h"
#include "shared.h"
#include "shared.cuh"

namespace StreamCompaction {
namespace Radix {
Common::PerformanceTimer& timer() { static Common::PerformanceTimer value; return value; }

__global__ void kernZeroBits(int n, int bit, int* flags, const int* input) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const unsigned key = static_cast<unsigned>(input[i]) ^ 0x80000000u;
        flags[i] = ((key >> bit) & 1u) == 0;
    }
}

__global__ void kernSplit(int n, int* output, const int* input, const int* flags, const int* indices) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const int totalZeros = indices[n - 1] + flags[n - 1];
        const int destination = flags[i] ? indices[i] : totalZeros + i - indices[i];
        output[destination] = input[i];
    }
}

void sort(int n, int* odata, const int* idata) {
    Common::validateSize(n);
    const int threads = Shared::blockSize();
    Common::DeviceBuffer a(n), b(n), flags(n), indices(n);
    Shared::Workspace workspace(n, 2 * threads);
    if (n) Common::cudaCheck(cudaMemcpy(a.get(), idata, size_t(n) * sizeof(int), cudaMemcpyHostToDevice), "Radix input");
    int* input = a.get();
    int* output = b.get();
    timer().startGpuTimer();
    if (n) {
        for (int bit = 0; bit < 32; ++bit) {
            kernZeroBits<<<Common::blocksFor(n, threads), threads>>>(n, bit, flags.get(), input);
            Shared::scanDevice(n, indices.get(), flags.get(), workspace, threads);
            kernSplit<<<Common::blocksFor(n, threads), threads>>>(n, output, input, flags.get(), indices.get());
            std::swap(input, output);
        }
    }
    timer().endGpuTimer();
    Common::cudaCheck(cudaGetLastError(), "Radix kernels");
    if (n) Common::cudaCheck(cudaMemcpy(odata, input, size_t(n) * sizeof(int), cudaMemcpyDeviceToHost), "Radix output");
}
}
}

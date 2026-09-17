#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"
#include "device_util.cuh"

namespace StreamCompaction {
    namespace Naive {
        namespace { int threads = 64; }
        void setBlockSize(int value) { Common::validateBlockSize(value); threads = value; }
        int blockSize() { return threads; }
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        // TODO: __global__
        __global__ void kernStep(int n, int offset, int* output, const int* input) {
            const int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < n) output[i] = input[i] + (i >= offset ? input[i - offset] : 0);
        }

        __global__ void kernExclusive(int n, int* output, const int* input) {
            const int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < n) output[i] = i ? input[i - 1] : 0;
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            Common::validateSize(n);
            Common::DeviceBuffer a(n), b(n);
            if (n) Common::cudaCheck(cudaMemcpy(a.get(), idata, size_t(n) * sizeof(int), cudaMemcpyHostToDevice), "Naive input");
            int* read = a.get();
            int* write = b.get();
            timer().startGpuTimer();
            // TODO
            if (n) {
                for (int d = 0; d < ilog2ceil(n); ++d) {
                    kernStep<<<Common::blocksFor(n, threads), threads>>>(n, 1 << d, write, read);
                    std::swap(read, write);
                }
                kernExclusive<<<Common::blocksFor(n, threads), threads>>>(n, write, read);
            }
            timer().endGpuTimer();
            Common::cudaCheck(cudaGetLastError(), "Naive scan kernels");
            if (n) Common::cudaCheck(cudaMemcpy(odata, write, size_t(n) * sizeof(int), cudaMemcpyDeviceToHost), "Naive output");
        }
    }
}

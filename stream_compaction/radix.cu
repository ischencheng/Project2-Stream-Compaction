#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"
#include "radix.h"

namespace StreamCompaction {
    namespace Radix {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        // 1 if bit b of x is 0, with the sign bit flipped so negative numbers come first
        __device__ int isZeroBit(int x, int b) {
            unsigned int key = (unsigned int)x ^ 0x80000000u;
            return ((key >> b) & 1u) == 0 ? 1 : 0;
        }

        __global__ void kernMapZeroBit(int n, int size, int b, int *e, const int *idata) {
            int index = blockIdx.x * blockDim.x + threadIdx.x;
            if (index < size) {
                // padding stays 0
                e[index] = index < n ? isZeroBit(idata[index], b) : 0;
            }
        }

        // split by bit b: zeros go to the front, ones after them, both keep their order
        __global__ void kernSplit(int n, int b, int *odata, const int *idata, const int *f) {
            int index = blockIdx.x * blockDim.x + threadIdx.x;
            if (index >= n) {
                return;
            }
            int totalZeros = f[n - 1] + isZeroBit(idata[n - 1], b);
            int x = idata[index];
            int dst = isZeroBit(x, b) ? f[index] : index - f[index] + totalZeros;
            odata[dst] = x;
        }

        /**
         * Sorts idata into odata from the lowest bit to the highest, one bit per pass.
         */
        void sort(int n, int *odata, const int *idata) {
            int size = 1 << ilog2ceil(n);
            int blockSize = Efficient::blockSize;
            int *dev_a, *dev_b, *dev_f;
            cudaMalloc((void**)&dev_a, n * sizeof(int));
            cudaMalloc((void**)&dev_b, n * sizeof(int));
            cudaMalloc((void**)&dev_f, size * sizeof(int));
            checkCUDAError("cudaMalloc failed");
            cudaMemcpy(dev_a, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            checkCUDAError("cudaMemcpy to device failed");

            timer().startGpuTimer();
            for (int b = 0; b < 32; b++) {
                kernMapZeroBit<<<(size + blockSize - 1) / blockSize, blockSize>>>(n, size, b, dev_f, dev_a);
                Efficient::scanOnDevice(size, dev_f);
                kernSplit<<<(n + blockSize - 1) / blockSize, blockSize>>>(n, b, dev_b, dev_a, dev_f);
                std::swap(dev_a, dev_b);
            }
            timer().endGpuTimer();
            checkCUDAError("radix sort failed");

            // 32 swaps, so the result is back in dev_a
            cudaMemcpy(odata, dev_a, n * sizeof(int), cudaMemcpyDeviceToHost);
            cudaFree(dev_a);
            cudaFree(dev_b);
            cudaFree(dev_f);
        }
    }
}

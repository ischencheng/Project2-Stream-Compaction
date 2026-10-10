#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        // TODO: __global__
        int blockSize = 128;

        // one level of the naive scan: add the element offset to the left
        __global__ void kernScanStep(int n, int offset, int *odata, const int *idata) {
            int index = blockIdx.x * blockDim.x + threadIdx.x;
            if (index >= n) {
                return;
            }
            if (index >= offset) {
                odata[index] = idata[index - offset] + idata[index];
            } else {
                odata[index] = idata[index];
            }
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            int *dev_a, *dev_b;
            cudaMalloc((void**)&dev_a, n * sizeof(int));
            cudaMalloc((void**)&dev_b, n * sizeof(int));
            checkCUDAError("cudaMalloc failed");
            // shift the input right by one, so the inclusive scan below gives an exclusive scan
            cudaMemset(dev_a, 0, sizeof(int));
            cudaMemcpy(dev_a + 1, idata, (n - 1) * sizeof(int), cudaMemcpyHostToDevice);
            checkCUDAError("cudaMemcpy to device failed");

            dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);
            timer().startGpuTimer();
            // TODO
            for (int d = 1; d <= ilog2ceil(n); d++) {
                kernScanStep<<<fullBlocksPerGrid, blockSize>>>(n, 1 << (d - 1), dev_b, dev_a);
                std::swap(dev_a, dev_b);
            }
            timer().endGpuTimer();
            checkCUDAError("kernScanStep failed");

            // after the last swap the result is in dev_a
            cudaMemcpy(odata, dev_a, n * sizeof(int), cudaMemcpyDeviceToHost);
            cudaFree(dev_a);
            cudaFree(dev_b);
        }
    }
}

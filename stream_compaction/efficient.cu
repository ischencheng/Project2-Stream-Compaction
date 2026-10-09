#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        int blockSize = 128;

        __global__ void kernUpSweep(int n, int stride, int *data) {
            int k = blockIdx.x * blockDim.x + threadIdx.x;
            if (k >= n || k % stride != 0) {
                return;
            }
            data[k + stride - 1] += data[k + stride / 2 - 1];
        }

        __global__ void kernDownSweep(int n, int stride, int *data) {
            int k = blockIdx.x * blockDim.x + threadIdx.x;
            if (k >= n || k % stride != 0) {
                return;
            }
            int left = data[k + stride / 2 - 1];
            data[k + stride / 2 - 1] = data[k + stride - 1];
            data[k + stride - 1] += left;
        }

        // in-place exclusive scan on the device, n must be a power of two
        void scanOnDevice(int n, int *dev_data) {
            dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);
            for (int d = 0; d < ilog2(n); d++) {
                kernUpSweep<<<fullBlocksPerGrid, blockSize>>>(n, 1 << (d + 1), dev_data);
            }
            // set the root to 0
            cudaMemset(dev_data + n - 1, 0, sizeof(int));
            for (int d = ilog2(n) - 1; d >= 0; d--) {
                kernDownSweep<<<fullBlocksPerGrid, blockSize>>>(n, 1 << (d + 1), dev_data);
            }
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            // round up to the next power of two and pad with zeros
            int size = 1 << ilog2ceil(n);
            int *dev_data;
            cudaMalloc((void**)&dev_data, size * sizeof(int));
            checkCUDAError("cudaMalloc failed");
            cudaMemset(dev_data, 0, size * sizeof(int));
            cudaMemcpy(dev_data, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            checkCUDAError("cudaMemcpy to device failed");

            timer().startGpuTimer();
            // TODO
            scanOnDevice(size, dev_data);
            timer().endGpuTimer();
            checkCUDAError("work-efficient scan failed");

            cudaMemcpy(odata, dev_data, n * sizeof(int), cudaMemcpyDeviceToHost);
            cudaFree(dev_data);
        }

        /**
         * Performs stream compaction on idata, storing the result into odata.
         * All zeroes are discarded.
         *
         * @param n      The number of elements in idata.
         * @param odata  The array into which to store elements.
         * @param idata  The array of elements to compact.
         * @returns      The number of elements remaining after compaction.
         */
        int compact(int n, int *odata, const int *idata) {
            timer().startGpuTimer();
            // TODO
            timer().endGpuTimer();
            return -1;
        }
    }
}

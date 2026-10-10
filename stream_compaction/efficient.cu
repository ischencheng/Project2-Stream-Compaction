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

        // thread k works on the k-th pair of this level, so no thread is idle
        __global__ void kernUpSweep(int count, int stride, int *data) {
            int k = blockIdx.x * blockDim.x + threadIdx.x;
            if (k >= count) {
                return;
            }
            int right = (k + 1) * stride - 1;
            data[right] += data[right - stride / 2];
        }

        __global__ void kernDownSweep(int count, int stride, int *data) {
            int k = blockIdx.x * blockDim.x + threadIdx.x;
            if (k >= count) {
                return;
            }
            int right = (k + 1) * stride - 1;
            int left = data[right - stride / 2];
            data[right - stride / 2] = data[right];
            data[right] += left;
        }

        // in-place exclusive scan on the device, n must be a power of two
        void scanOnDevice(int n, int *dev_data) {
            for (int d = 0; d < ilog2(n); d++) {
                // only n / 2^(d+1) pairs at this level
                int count = n >> (d + 1);
                dim3 blocks((count + blockSize - 1) / blockSize);
                kernUpSweep<<<blocks, blockSize>>>(count, 1 << (d + 1), dev_data);
            }
            // set the root to 0
            cudaMemset(dev_data + n - 1, 0, sizeof(int));
            for (int d = ilog2(n) - 1; d >= 0; d--) {
                int count = n >> (d + 1);
                dim3 blocks((count + blockSize - 1) / blockSize);
                kernDownSweep<<<blocks, blockSize>>>(count, 1 << (d + 1), dev_data);
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
            int size = 1 << ilog2ceil(n);
            int *dev_idata, *dev_odata, *dev_bools, *dev_indices;
            cudaMalloc((void**)&dev_idata, n * sizeof(int));
            cudaMalloc((void**)&dev_odata, n * sizeof(int));
            cudaMalloc((void**)&dev_bools, n * sizeof(int));
            cudaMalloc((void**)&dev_indices, size * sizeof(int));
            checkCUDAError("cudaMalloc failed");
            cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            cudaMemset(dev_indices, 0, size * sizeof(int));
            checkCUDAError("cudaMemcpy to device failed");

            dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);
            timer().startGpuTimer();
            // TODO
            Common::kernMapToBoolean<<<fullBlocksPerGrid, blockSize>>>(n, dev_bools, dev_idata);
            // the scan is in place, so scan a copy and keep the bools for scatter
            cudaMemcpy(dev_indices, dev_bools, n * sizeof(int), cudaMemcpyDeviceToDevice);
            scanOnDevice(size, dev_indices);
            Common::kernScatter<<<fullBlocksPerGrid, blockSize>>>(n, dev_odata, dev_idata, dev_bools, dev_indices);
            timer().endGpuTimer();
            checkCUDAError("work-efficient compact failed");

            // count = last index + last bool
            int lastIndex, lastBool;
            cudaMemcpy(&lastIndex, dev_indices + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(&lastBool, dev_bools + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
            int count = lastIndex + lastBool;
            cudaMemcpy(odata, dev_odata, count * sizeof(int), cudaMemcpyDeviceToHost);

            cudaFree(dev_idata);
            cudaFree(dev_odata);
            cudaFree(dev_bools);
            cudaFree(dev_indices);
            return count;
        }
    }
}

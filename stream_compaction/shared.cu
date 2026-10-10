#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "shared.h"

namespace StreamCompaction {
    namespace Shared {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        int blockSize = 128;
        bool avoidBankConflicts = true;

        // skip one slot every 32 ints, so the threads of a warp hit different banks
        template <bool Pad>
        __device__ int bank(int i) {
            return Pad ? i + (i >> 5) : i;
        }

        // each block scans a tile of 2 * blockDim.x elements and writes the tile sum to sums
        template <bool Pad>
        __global__ void kernScanTile(int n, int *odata, const int *idata, int *sums) {
            extern __shared__ int temp[];
            int t = threadIdx.x;
            int tile = 2 * blockDim.x;
            int base = blockIdx.x * tile;
            int ai = t;
            int bi = t + blockDim.x;
            temp[bank<Pad>(ai)] = base + ai < n ? idata[base + ai] : 0;
            temp[bank<Pad>(bi)] = base + bi < n ? idata[base + bi] : 0;

            int stride = 1;
            for (int count = blockDim.x; count > 0; count >>= 1) {
                __syncthreads();
                if (t < count) {
                    int left = stride * (2 * t + 1) - 1;
                    int right = stride * (2 * t + 2) - 1;
                    temp[bank<Pad>(right)] += temp[bank<Pad>(left)];
                }
                stride <<= 1;
            }
            if (t == 0) {
                if (sums) sums[blockIdx.x] = temp[bank<Pad>(tile - 1)];
                temp[bank<Pad>(tile - 1)] = 0;
            }
            for (int count = 1; count < tile; count <<= 1) {
                stride >>= 1;
                __syncthreads();
                if (t < count) {
                    int left = stride * (2 * t + 1) - 1;
                    int right = stride * (2 * t + 2) - 1;
                    int x = temp[bank<Pad>(left)];
                    temp[bank<Pad>(left)] = temp[bank<Pad>(right)];
                    temp[bank<Pad>(right)] += x;
                }
            }
            __syncthreads();
            if (base + ai < n) odata[base + ai] = temp[bank<Pad>(ai)];
            if (base + bi < n) odata[base + bi] = temp[bank<Pad>(bi)];
        }

        // each block scans a tile of blockDim.x elements with two buffers
        __global__ void kernNaiveScanTile(int n, int *odata, const int *idata, int *sums) {
            extern __shared__ int temp[];
            int t = threadIdx.x;
            int index = blockIdx.x * blockDim.x + t;
            int pout = 0;
            temp[t] = index < n ? idata[index] : 0;
            __syncthreads();
            for (int offset = 1; offset < blockDim.x; offset <<= 1) {
                pout = 1 - pout;
                int *in = temp + (1 - pout) * blockDim.x;
                int x = in[t];
                if (t >= offset) x += in[t - offset];
                temp[pout * blockDim.x + t] = x;
                __syncthreads();
            }
            // inclusive scan is in buffer pout, shift it for the exclusive one
            int *result = temp + pout * blockDim.x;
            if (sums && t == blockDim.x - 1) sums[blockIdx.x] = result[t];
            if (index < n) odata[index] = t > 0 ? result[t - 1] : 0;
        }

        __global__ void kernAddOffsets(int n, int tile, int *data, const int *offsets) {
            int offset = offsets[blockIdx.x];
            for (int i = threadIdx.x; i < tile; i += blockDim.x) {
                int index = blockIdx.x * tile + i;
                if (index < n) data[index] += offset;
            }
        }

        // in-place scan, sums needs room for the tile sums of every level
        void scanOnDevice(int n, int *data, int *sums, bool naive) {
            int tile = naive ? blockSize : 2 * blockSize;
            int blocks = (n + tile - 1) / tile;
            int *tileSums = blocks > 1 ? sums : nullptr;
            if (naive) {
                kernNaiveScanTile<<<blocks, blockSize, 2 * blockSize * sizeof(int)>>>(n, data, data, tileSums);
            } else if (avoidBankConflicts) {
                kernScanTile<true><<<blocks, blockSize, (tile + tile / 32) * sizeof(int)>>>(n, data, data, tileSums);
            } else {
                kernScanTile<false><<<blocks, blockSize, tile * sizeof(int)>>>(n, data, data, tileSums);
            }
            if (blocks > 1) {
                // scan the tile sums, then add them to every tile
                scanOnDevice(blocks, sums, sums + blocks, naive);
                kernAddOffsets<<<blocks, blockSize>>>(n, tile, data, sums);
            }
        }

        void scanHost(int n, int *odata, const int *idata, bool naive) {
            int tile = naive ? blockSize : 2 * blockSize;
            int sumsSize = 1;
            for (int m = n; m > tile; m = (m + tile - 1) / tile) {
                sumsSize += (m + tile - 1) / tile;
            }
            int *dev_data, *dev_sums;
            cudaMalloc((void**)&dev_data, n * sizeof(int));
            cudaMalloc((void**)&dev_sums, sumsSize * sizeof(int));
            checkCUDAError("cudaMalloc failed");
            cudaMemcpy(dev_data, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            checkCUDAError("cudaMemcpy to device failed");

            timer().startGpuTimer();
            scanOnDevice(n, dev_data, dev_sums, naive);
            timer().endGpuTimer();
            checkCUDAError("shared memory scan failed");

            cudaMemcpy(odata, dev_data, n * sizeof(int), cudaMemcpyDeviceToHost);
            cudaFree(dev_data);
            cudaFree(dev_sums);
        }

        void scan(int n, int *odata, const int *idata) {
            scanHost(n, odata, idata, false);
        }

        void scanNaive(int n, int *odata, const int *idata) {
            scanHost(n, odata, idata, true);
        }
    }
}

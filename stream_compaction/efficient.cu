#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"
#include "device_util.cuh"

namespace StreamCompaction {
    namespace Efficient {
        namespace { int threads = 64; }
        void setBlockSize(int value) { Common::validateBlockSize(value); threads = value; }
        int blockSize() { return threads; }
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        template<bool CompactThreads>
        __global__ void kernUpsweep(int padded, int stride, int* data) {
            const int i = blockIdx.x * blockDim.x + threadIdx.x;
            const int active = padded / stride;
            if (CompactThreads ? i >= active : (i >= padded || (i + 1) % stride != 0)) return;
            const int right = CompactThreads ? (i + 1) * stride - 1 : i;
            data[right] += data[right - stride / 2];
        }

        template<bool CompactThreads>
        __global__ void kernDownsweep(int padded, int stride, int* data) {
            const int i = blockIdx.x * blockDim.x + threadIdx.x;
            const int active = padded / stride;
            if (CompactThreads ? i >= active : (i >= padded || (i + 1) % stride != 0)) return;
            const int right = CompactThreads ? (i + 1) * stride - 1 : i;
            const int left = right - stride / 2;
            const int value = data[left];
            data[left] = data[right];
            data[right] += value;
        }

        __global__ void kernClearRoot(int padded, int* data) { data[padded - 1] = 0; }

        __global__ void kernPrepareFlags(int n, int padded, int* indices, const int* flags) {
            const int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < padded) indices[i] = i < n ? flags[i] : 0;
        }

        template<bool CompactThreads>
        void scanDevice(int padded, int* data) {
            const int levels = ilog2ceil(padded);
            for (int d = 1; d <= levels; ++d) {
                const int stride = 1 << d;
                const int work = CompactThreads ? padded / stride : padded;
                kernUpsweep<CompactThreads><<<Common::blocksFor(work, threads), threads>>>(padded, stride, data);
            }
            kernClearRoot<<<1, 1>>>(padded, data);
            for (int d = levels; d >= 1; --d) {
                const int stride = 1 << d;
                const int work = CompactThreads ? padded / stride : padded;
                kernDownsweep<CompactThreads><<<Common::blocksFor(work, threads), threads>>>(padded, stride, data);
            }
        }

        template<bool CompactThreads>
        void scanHost(int n, int* odata, const int* idata) {
            Common::validateSize(n);
            const int padded = n ? 1 << ilog2ceil(n) : 0;
            Common::DeviceBuffer data(padded);
            if (n) {
                Common::cudaCheck(cudaMemset(data.get(), 0, size_t(padded) * sizeof(int)), "Clear padding");
                Common::cudaCheck(cudaMemcpy(data.get(), idata, size_t(n) * sizeof(int), cudaMemcpyHostToDevice), "Efficient input");
            }
            timer().startGpuTimer();
            if (n) scanDevice<CompactThreads>(padded, data.get());
            timer().endGpuTimer();
            Common::cudaCheck(cudaGetLastError(), "Efficient scan kernels");
            if (n) Common::cudaCheck(cudaMemcpy(odata, data.get(), size_t(n) * sizeof(int), cudaMemcpyDeviceToHost), "Efficient output");
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            // TODO
            scanHost<true>(n, odata, idata);
        }

        void scanUnoptimized(int n, int* odata, const int* idata) { scanHost<false>(n, odata, idata); }

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
            Common::validateSize(n);
            const int padded = n ? 1 << ilog2ceil(n) : 0;
            Common::DeviceBuffer input(n), output(n), flags(n), indices(padded);
            if (n) Common::cudaCheck(cudaMemcpy(input.get(), idata, size_t(n) * sizeof(int), cudaMemcpyHostToDevice), "Compaction input");
            timer().startGpuTimer();
            // TODO
            if (n) {
                Common::kernMapToBoolean<<<Common::blocksFor(n, threads), threads>>>(n, flags.get(), input.get());
                kernPrepareFlags<<<Common::blocksFor(padded, threads), threads>>>(n, padded, indices.get(), flags.get());
                scanDevice<true>(padded, indices.get());
                Common::kernScatter<<<Common::blocksFor(n, threads), threads>>>(n, output.get(), input.get(), flags.get(), indices.get());
            }
            timer().endGpuTimer();
            Common::cudaCheck(cudaGetLastError(), "Compaction kernels");
            int count = 0;
            if (n) {
                Common::cudaCheck(cudaMemcpy(&count, indices.get() + n - 1, sizeof(int), cudaMemcpyDeviceToHost), "Compaction count");
                count += idata[n - 1] != 0;
                if (count) Common::cudaCheck(cudaMemcpy(odata, output.get(), size_t(count) * sizeof(int), cudaMemcpyDeviceToHost), "Compaction output");
            }
            return count;
        }
    }
}

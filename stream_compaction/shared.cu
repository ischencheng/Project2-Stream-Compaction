#include "shared.h"
#include "shared.cuh"

namespace StreamCompaction {
namespace Shared {
namespace { int threads = 128; }
void setBlockSize(int value) { Common::validateBlockSize(value); threads = value; }
int blockSize() { return threads; }
Common::PerformanceTimer& timer() { static Common::PerformanceTimer value; return value; }

template<bool Padded>
__device__ int address(int i) {
    // Modern GPUs have 32 banks. The second term covers tiles larger than 1024.
    return Padded ? i + (i >> 5) + (i >> 10) : i;
}

template<bool Padded>
__global__ void kernBlelloch(int n, int* output, const int* input, int* sums) {
    extern __shared__ int data[];
    const int t = threadIdx.x;
    const int tile = 2 * blockDim.x;
    const int base = blockIdx.x * tile;
    const int a = t, b = t + blockDim.x;
    data[address<Padded>(a)] = base + a < n ? input[base + a] : 0;
    data[address<Padded>(b)] = base + b < n ? input[base + b] : 0;
    for (int stride = 2; stride <= tile; stride <<= 1) {
        __syncthreads();
        if (t < tile / stride) {
            const int right = (t + 1) * stride - 1;
            data[address<Padded>(right)] += data[address<Padded>(right - stride / 2)];
        }
    }
    __syncthreads();
    if (t == 0) {
        if (sums) sums[blockIdx.x] = data[address<Padded>(tile - 1)];
        data[address<Padded>(tile - 1)] = 0;
    }
    for (int stride = tile; stride >= 2; stride >>= 1) {
        __syncthreads();
        if (t < tile / stride) {
            const int right = address<Padded>((t + 1) * stride - 1);
            const int left = address<Padded>((t + 1) * stride - 1 - stride / 2);
            const int value = data[left];
            data[left] = data[right];
            data[right] += value;
        }
    }
    __syncthreads();
    if (base + a < n) output[base + a] = data[address<Padded>(a)];
    if (base + b < n) output[base + b] = data[address<Padded>(b)];
}

__global__ void kernNaiveShared(int n, int* output, const int* input, int* sums) {
    extern __shared__ int data[];
    const int t = threadIdx.x;
    const int tile = blockDim.x;
    const int i = blockIdx.x * tile + t;
    int* read = data;
    int* write = data + tile;
    read[t] = i < n ? input[i] : 0;
    __syncthreads();
    for (int offset = 1; offset < tile; offset <<= 1) {
        write[t] = read[t] + (t >= offset ? read[t - offset] : 0);
        __syncthreads();
        int* previous = read;
        read = write;
        write = previous;
    }
    if (sums && t == tile - 1) sums[blockIdx.x] = read[t];
    if (i < n) output[i] = t ? read[t - 1] : 0;
}

template<bool Naive>
__global__ void kernAddOffsets(int n, int* data, const int* offsets) {
    const int tile = (Naive ? 1 : 2) * blockDim.x;
    const int i = blockIdx.x * tile + threadIdx.x;
    const int offset = offsets[blockIdx.x];
    if (i < n) data[i] += offset;
    if (!Naive && i + blockDim.x < n) data[i + blockDim.x] += offset;
}

template<bool Naive, bool Padded>
void scanRecursive(int n, int* output, const int* input, Workspace& workspace, int block,
                   int level = 0) {
    const int tile = block * (Naive ? 1 : 2);
    const int blocks = Common::blocksFor(n, tile);
    int* sums = blocks > 1 ? workspace.sums[level]->get() : nullptr;
    if (Naive) {
        kernNaiveShared<<<blocks, block, 2 * block * sizeof(int)>>>(n, output, input, sums);
    } else {
        const int slots = Padded ? tile + (tile >> 5) + (tile >> 10) : tile;
        kernBlelloch<Padded><<<blocks, block, slots * sizeof(int)>>>(n, output, input, sums);
    }
    if (blocks > 1) {
        int* offsets = workspace.offsets[level]->get();
        scanRecursive<Naive, Padded>(blocks, offsets, sums, workspace, block, level + 1);
        kernAddOffsets<Naive><<<blocks, block>>>(n, output, offsets);
    }
}

void scanDevice(int n, int* output, const int* input, Workspace& workspace, int block) {
    if (n) scanRecursive<false, true>(n, output, input, workspace, block);
}

template<bool Naive, bool Padded>
void scanHost(int n, int* odata, const int* idata) {
    Common::validateSize(n);
    Common::DeviceBuffer input(n), output(n);
    Workspace workspace(n, threads * (Naive ? 1 : 2));
    if (n) Common::cudaCheck(cudaMemcpy(input.get(), idata, size_t(n) * sizeof(int), cudaMemcpyHostToDevice), "Shared input");
    timer().startGpuTimer();
    if (n) scanRecursive<Naive, Padded>(n, output.get(), input.get(), workspace, threads);
    timer().endGpuTimer();
    Common::cudaCheck(cudaGetLastError(), "Shared scan kernels");
    if (n) Common::cudaCheck(cudaMemcpy(odata, output.get(), size_t(n) * sizeof(int), cudaMemcpyDeviceToHost), "Shared output");
}

void scan(int n, int* odata, const int* idata) { scanHost<false, true>(n, odata, idata); }
void scanNaive(int n, int* odata, const int* idata) { scanHost<true, false>(n, odata, idata); }
void scanUnpadded(int n, int* odata, const int* idata) { scanHost<false, false>(n, odata, idata); }

int compact(int n, int* odata, const int* idata) {
    Common::validateSize(n);
    Common::DeviceBuffer input(n), output(n), flags(n), indices(n);
    Workspace workspace(n, 2 * threads);
    if (n) Common::cudaCheck(cudaMemcpy(input.get(), idata, size_t(n) * sizeof(int), cudaMemcpyHostToDevice), "Shared compaction input");
    timer().startGpuTimer();
    if (n) {
        Common::kernMapToBoolean<<<Common::blocksFor(n, threads), threads>>>(n, flags.get(), input.get());
        scanDevice(n, indices.get(), flags.get(), workspace, threads);
        Common::kernScatter<<<Common::blocksFor(n, threads), threads>>>(n, output.get(), input.get(), flags.get(), indices.get());
    }
    timer().endGpuTimer();
    Common::cudaCheck(cudaGetLastError(), "Shared compaction kernels");
    int count = 0;
    if (n) {
        Common::cudaCheck(cudaMemcpy(&count, indices.get() + n - 1, sizeof(int), cudaMemcpyDeviceToHost), "Shared compaction count");
        count += idata[n - 1] != 0;
        if (count) Common::cudaCheck(cudaMemcpy(odata, output.get(), size_t(count) * sizeof(int), cudaMemcpyDeviceToHost), "Shared compaction output");
    }
    return count;
}

void printOccupancy() {
    cudaDeviceProp properties{};
    Common::cudaCheck(cudaGetDeviceProperties(&properties, 0), "Device properties");
    printf("GPU: %s; SM %d.%d; %d SMs; %zu bytes shared/SM; %d threads/SM\n", properties.name,
           properties.major, properties.minor, properties.multiProcessorCount,
           properties.sharedMemPerMultiprocessor, properties.maxThreadsPerMultiProcessor);
    printf("block,shared_bytes,active_blocks_per_sm,theoretical_occupancy\n");
    for (int block : {32, 64, 128, 256, 512, 1024}) {
        const size_t bytes = (2 * block + (2 * block >> 5) + (2 * block >> 10)) * sizeof(int);
        int active = 0;
        Common::cudaCheck(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active, kernBlelloch<true>, block, bytes), "Occupancy query");
        printf("%d,%zu,%d,%.3f\n", block, bytes, active, double(active * block) / properties.maxThreadsPerMultiProcessor);
    }
}
}
}

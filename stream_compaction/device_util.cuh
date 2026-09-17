#pragma once

#include "common.h"
#include <string>

namespace StreamCompaction {
namespace Common {

inline void cudaCheck(cudaError_t result, const char* operation) {
    if (result != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(result));
    }
}

inline void validateSize(int n) {
    // Keep padded tree indices representable as signed 32-bit integers.
    if (n < 0 || n > (1 << 30)) {
        throw std::invalid_argument("Size must be between 0 and 2^30.");
    }
}

inline void validateBlockSize(int blockSize) {
    if (blockSize < 32 || blockSize > 1024 || (blockSize & (blockSize - 1))) {
        throw std::invalid_argument("Block size must be a power of two from 32 to 1024.");
    }
}

inline int blocksFor(int n, int blockSize) {
    return (n + blockSize - 1) / blockSize;
}

class DeviceBuffer {
public:
    explicit DeviceBuffer(int n) {
        if (n > 0) cudaCheck(cudaMalloc(&data_, size_t(n) * sizeof(int)), "cudaMalloc");
    }
    ~DeviceBuffer() { if (data_) cudaFree(data_); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    int* get() const { return data_; }
private:
    int* data_ = nullptr;
};

}
}

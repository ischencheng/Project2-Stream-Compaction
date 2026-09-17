#pragma once
#include "device_util.cuh"
#include <memory>
#include <vector>

namespace StreamCompaction {
namespace Shared {
// Allocate every recursion level before recording the start event.
struct Workspace {
    explicit Workspace(int n, int tileSize) {
        while (n > tileSize) {
            n = Common::blocksFor(n, tileSize);
            sums.emplace_back(new Common::DeviceBuffer(n));
            offsets.emplace_back(new Common::DeviceBuffer(n));
        }
    }
    std::vector<std::unique_ptr<Common::DeviceBuffer>> sums, offsets;
};

// Device-only Blelloch scan reused by compaction and all 32 radix passes.
void scanDevice(int n, int* output, const int* input, Workspace& workspace, int threads);
}
}

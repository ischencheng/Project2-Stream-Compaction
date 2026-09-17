#include "benchmark.h"
#include <stream_compaction/cpu.h>
#include <stream_compaction/naive.h>
#include <stream_compaction/efficient.h>
#include <stream_compaction/thrust.h>
#include <stream_compaction/shared.h>
#include <stream_compaction/radix.h>
#include <cuda_profiler_api.h>
#include <algorithm>
#include <chrono>
#include <fstream>
#include <functional>
#include <iomanip>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

namespace {
using namespace StreamCompaction;
struct Entry {
    const char* name;
    const char* kind;
    std::function<void(int, int*, const int*)> run;
    Common::PerformanceTimer& (*timer)();
    void (*setBlock)(int);
    int block;
    bool gpu;
};

std::vector<Entry> entries() {
    return {
        {"CPU", "scan", CPU::scan, CPU::timer, nullptr, 0, false},
        {"Naive", "scan", Naive::scan, Naive::timer, Naive::setBlockSize, 64, true},
        {"Efficient", "scan", Efficient::scan, Efficient::timer, Efficient::setBlockSize, 64, true},
        {"Thrust", "scan", Thrust::scan, Thrust::timer, nullptr, 0, true},
        {"Efficient unoptimized", "scan", Efficient::scanUnoptimized, Efficient::timer, Efficient::setBlockSize, 256, true},
        {"Shared naive", "scan", Shared::scanNaive, Shared::timer, Shared::setBlockSize, 128, true},
        {"Shared Blelloch", "scan", Shared::scan, Shared::timer, Shared::setBlockSize, 128, true},
        {"Shared unpadded", "scan", Shared::scanUnpadded, Shared::timer, Shared::setBlockSize, 64, true},
        {"CPU direct", "compact", CPU::compactWithoutScan, CPU::timer, nullptr, 0, false},
        {"CPU scan", "compact", CPU::compactWithScan, CPU::timer, nullptr, 0, false},
        {"Efficient compact", "compact", Efficient::compact, Efficient::timer, Efficient::setBlockSize, 64, true},
        {"Shared compact", "compact", Shared::compact, Shared::timer, Shared::setBlockSize, 128, true}
    };
}
}

int runBenchmark(const char* path, bool tune) {
    const int warmup = 3, repeats = 15;
    std::ofstream csv(path);
    if (!csv) throw std::runtime_error("Cannot open benchmark output.");
    csv << "kind,algorithm,n,block,repeat,algorithm_ms,wall_ms\n" << std::setprecision(9);
    std::vector<int> sizes = tune ? std::vector<int>{10000, 1048576} :
        std::vector<int>{256, 1024, 10000, 65536, 262144, 1000000, 1048576, 4194304, 16777216};
    std::mt19937 rng(5650);
    for (int n : sizes) {
        std::vector<int> input(n), output(n), expected(n), compacted;
        for (int& value : input) value = static_cast<int>(rng() % 4);
        std::exclusive_scan(input.begin(), input.end(), expected.begin(), 0);
        std::copy_if(input.begin(), input.end(), std::back_inserter(compacted), [](int x) { return x != 0; });
        auto algorithms = entries();
        // Deterministic shuffled order reduces systematic thermal/order bias.
        std::shuffle(algorithms.begin(), algorithms.end(), rng);
        for (auto& entry : algorithms) {
            if (tune && (!entry.setBlock || std::string(entry.kind) != "scan")) continue;
            auto blocks = tune ? std::vector<int>{32, 64, 128, 256, 512, 1024} : std::vector<int>{entry.block};
            std::shuffle(blocks.begin(), blocks.end(), rng);
            for (int block : blocks) {
                if (entry.setBlock) entry.setBlock(block);
                std::vector<float> measurements;
                for (int repeat = -warmup; repeat < repeats; ++repeat) {
                    const auto start = std::chrono::steady_clock::now();
                    entry.run(n, output.data(), input.data());
                    const auto end = std::chrono::steady_clock::now();
                    const float elapsed = entry.gpu ? entry.timer().getGpuElapsedTimeForPreviousOperation() :
                                                     entry.timer().getCpuElapsedTimeForPreviousOperation();
                    if (repeat >= 0) {
                        const double wall = std::chrono::duration<double, std::milli>(end - start).count();
                        csv << entry.kind << ',' << entry.name << ',' << n << ',' << block << ',' << repeat
                            << ',' << elapsed << ',' << wall << '\n';
                        measurements.push_back(elapsed);
                    }
                }
                const auto& reference = std::string(entry.kind) == "scan" ? expected : compacted;
                if (!std::equal(reference.begin(), reference.end(), output.begin())) {
                    throw std::runtime_error(std::string("Benchmark validation failed: ") + entry.name);
                }
                std::sort(measurements.begin(), measurements.end());
                printf("%s: n=%d block=%d median=%.6f ms\n", entry.name, n, block, measurements[repeats / 2]);
                fflush(stdout);
            }
        }
    }
    if (!csv) throw std::runtime_error("Could not finish writing benchmark output.");
    printf("Saved %s (%d warmups, %d samples per configuration).\n", path, warmup, repeats);
    return 0;
}

int runThrustProfile() {
    const int n = 1 << 20;
    std::vector<int> input(n, 1), output(n);
    for (int i = 0; i < 3; ++i) StreamCompaction::Thrust::scan(n, output.data(), input.data());
    if (cudaProfilerStart() != cudaSuccess) throw std::runtime_error("Cannot start profiler capture.");
    StreamCompaction::Thrust::scan(n, output.data(), input.data());
    if (cudaProfilerStop() != cudaSuccess) throw std::runtime_error("Cannot stop profiler capture.");
    for (int i = 0; i < n; ++i) if (output[i] != i) throw std::runtime_error("Profile validation failed.");
    printf("PASS: profiled one warmed Thrust scan of %d integers.\n", n);
    return 0;
}

int runRadixExample() {
    const std::vector<int> input{4, -7, 2, 6, 3, -5, 1, 0};
    std::vector<int> output(input.size());
    StreamCompaction::Radix::sort(static_cast<int>(input.size()), output.data(), input.data());
    printf("Radix input: "); for (int value : input) printf("%d ", value); printf("\n");
    printf("Radix output: "); for (int value : output) printf("%d ", value); printf("\n");
    return std::is_sorted(output.begin(), output.end()) ? 0 : 1;
}

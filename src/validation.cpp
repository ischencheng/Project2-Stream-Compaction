#include "validation.h"
#include <stream_compaction/cpu.h>
#include <stream_compaction/naive.h>
#include <stream_compaction/efficient.h>
#include <stream_compaction/thrust.h>
#include <stream_compaction/shared.h>
#include <stream_compaction/radix.h>
#include <climits>
#include <algorithm>
#include <cstdio>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using Scan = void (*)(int, int*, const int*);
using Compact = int (*)(int, int*, const int*);
constexpr int guard = 0x12345678;

void checkScan(const std::vector<int>& input, const char* name, Scan scan) {
    const int n = static_cast<int>(input.size());
    std::vector<int> expected(n), output(n + 2, guard);
    std::exclusive_scan(input.begin(), input.end(), expected.begin(), 0);
    scan(n, output.data() + 1, input.data());
    if (output.front() != guard || output.back() != guard ||
        !std::equal(expected.begin(), expected.end(), output.begin() + 1)) {
        throw std::runtime_error(std::string(name) + " scan mismatch, n=" + std::to_string(n));
    }
}

void checkCompact(const std::vector<int>& input, const char* name, Compact compact) {
    const int n = static_cast<int>(input.size());
    std::vector<int> expected, output(n + 2, guard);
    std::copy_if(input.begin(), input.end(), std::back_inserter(expected), [](int x) { return x != 0; });
    const int count = compact(n, output.data() + 1, input.data());
    if (count != static_cast<int>(expected.size()) || output.front() != guard || output.back() != guard ||
        !std::equal(expected.begin(), expected.end(), output.begin() + 1)) {
        throw std::runtime_error(std::string(name) + " compaction mismatch, n=" + std::to_string(n));
    }
}

void checkSort(const std::vector<int>& input) {
    auto expected = input;
    std::stable_sort(expected.begin(), expected.end());
    std::vector<int> output(input.size() + 2, guard);
    StreamCompaction::Radix::sort(static_cast<int>(input.size()), output.data() + 1, input.data());
    if (output.front() != guard || output.back() != guard ||
        !std::equal(expected.begin(), expected.end(), output.begin() + 1)) {
        throw std::runtime_error("Radix mismatch, n=" + std::to_string(input.size()));
    }
}
}

int runValidation(bool cpuOnly, bool quick) {
    std::vector<int> sizes = {0, 1, 2, 3, 7, 31, 32, 33, 63, 64, 65, 127, 128, 129,
                              253, 256, 257, 511, 512, 513, 1023, 1024, 1025, 10000};
    if (!quick) sizes.insert(sizes.end(), {65535, 65536, 65537, 1000000, 1048576, 1048579});
    else sizes = {0, 1, 3, 33, 257, 1025, 10000, 262145};
    std::mt19937 rng(5650);
    int cases = 0;
    auto check = [&](const std::vector<int>& input) {
        const auto original = input;
        checkScan(input, "CPU", StreamCompaction::CPU::scan);
        checkCompact(input, "CPU direct", StreamCompaction::CPU::compactWithoutScan);
        checkCompact(input, "CPU scan", StreamCompaction::CPU::compactWithScan);
        if (!cpuOnly) {
            checkScan(input, "Naive", StreamCompaction::Naive::scan);
            checkScan(input, "Efficient", StreamCompaction::Efficient::scan);
            checkScan(input, "Thrust", StreamCompaction::Thrust::scan);
            checkScan(input, "Efficient unoptimized", StreamCompaction::Efficient::scanUnoptimized);
            checkScan(input, "Shared naive", StreamCompaction::Shared::scanNaive);
            checkScan(input, "Shared Blelloch", StreamCompaction::Shared::scan);
            checkScan(input, "Shared unpadded", StreamCompaction::Shared::scanUnpadded);
            checkCompact(input, "Efficient", StreamCompaction::Efficient::compact);
            checkCompact(input, "Shared", StreamCompaction::Shared::compact);
            checkSort(input);
        }
        if (input != original) throw std::runtime_error("Input was modified.");
        ++cases;
    };
    check({1, 5, 0, 1, 2, 0, 3});
    for (int n : sizes) {
        for (int pattern = 0; pattern < 5; ++pattern) {
            std::vector<int> input(n);
            for (int i = 0; i < n; ++i) {
                if (pattern == 1) input[i] = 1;
                if (pattern == 2) input[i] = i % 2 ? -3 : 0;
                if (pattern == 3) input[i] = static_cast<int>(rng() % 19) - 9;
                if (pattern == 4) input[i] = i == n - 1 ? 7 : 0;
            }
            check(input);
        }
    }
    if (!cpuOnly) {
        // Integer extremes are valid sort inputs even when their scan could overflow.
        checkSort({INT_MAX, 0, INT_MIN, -1, 1, INT_MIN, INT_MAX, -7, 7});
        std::vector<int> signedKeys(quick ? 1025 : 10000);
        for (int& value : signedKeys) {
            value = static_cast<int>(rng() & 0x7fffffffu);
            if (rng() & 1u) value = -value;
        }
        checkSort(signedKeys);
        checkSort({});
        for (int block : {32, 64, 128, 256, 512, 1024}) {
            StreamCompaction::Naive::setBlockSize(block);
            StreamCompaction::Efficient::setBlockSize(block);
            StreamCompaction::Shared::setBlockSize(block);
            for (int n : {block - 1, block, block + 1, 2 * block + 1}) {
                std::vector<int> input(n);
                for (int& value : input) value = static_cast<int>(rng() % 7) - 3;
                check(input);
            }
        }
        StreamCompaction::Naive::setBlockSize(64);
        StreamCompaction::Efficient::setBlockSize(64);
        StreamCompaction::Shared::setBlockSize(128);
        printf("PASS: radix signed extremes and random full-range keys; block sizes 32 through 1024.\n");
    }
    printf("PASS: %d input cases; independent std::exclusive_scan and std::copy_if references.\n", cases);
    printf("PASS: empty/singleton, zero/one/signed/random/tail-only data, block boundaries, non-power-of-two sizes.\n");
    printf("PASS: output guards and input preservation (%s).\n", cpuOnly ? "CPU only" : "CPU and GPU");
    return 0;
}

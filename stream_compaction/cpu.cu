#include <cstdio>
#include <vector>
#include "cpu.h"

#include "common.h"

namespace StreamCompaction {
    namespace CPU {
        namespace {
            void exclusiveScan(int n, int* odata, const int* idata) {
                int sum = 0;
                for (int i = 0; i < n; ++i) {
                    const int value = idata[i];
                    odata[i] = sum;
                    sum += value;
                }
            }
            void validateSize(int n) {
                if (n < 0 || n > (1 << 30)) throw std::invalid_argument("Invalid size.");
            }
        }
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        /**
         * CPU scan (prefix sum).
         * For performance analysis, this is supposed to be a simple for loop.
         * (Optional) For better understanding before starting moving to GPU, you can simulate your GPU scan in this function first.
         */
        void scan(int n, int *odata, const int *idata) {
            validateSize(n);
            timer().startCpuTimer();
            // TODO
            exclusiveScan(n, odata, idata);
            timer().endCpuTimer();
        }

        /**
         * CPU stream compaction without using the scan function.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithoutScan(int n, int *odata, const int *idata) {
            validateSize(n);
            int count = 0;
            timer().startCpuTimer();
            // TODO
            for (int i = 0; i < n; ++i) {
                if (idata[i] != 0) odata[count++] = idata[i];
            }
            timer().endCpuTimer();
            return count;
        }

        /**
         * CPU stream compaction using scan and scatter, like the parallel version.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithScan(int n, int *odata, const int *idata) {
            validateSize(n);
            std::vector<int> flags(n), indices(n);
            timer().startCpuTimer();
            // TODO
            for (int i = 0; i < n; ++i) flags[i] = idata[i] != 0;
            // Share the scan core without nesting the public performance timer.
            exclusiveScan(n, indices.data(), flags.data());
            for (int i = 0; i < n; ++i) {
                if (flags[i]) odata[indices[i]] = idata[i];
            }
            const int count = n ? indices[n - 1] + flags[n - 1] : 0;
            timer().endCpuTimer();
            return count;
        }
    }
}

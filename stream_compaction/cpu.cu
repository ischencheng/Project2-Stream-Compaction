#include <cstdio>
#include "cpu.h"

#include "common.h"

namespace StreamCompaction {
    namespace CPU {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        // scan without the timer, so compactWithScan can call it too
        void exclusiveScan(int n, int *odata, const int *idata) {
            int sum = 0;
            for (int i = 0; i < n; i++) {
                int x = idata[i];
                odata[i] = sum;
                sum += x;
            }
        }

        /**
         * CPU scan (prefix sum).
         * For performance analysis, this is supposed to be a simple for loop.
         * (Optional) For better understanding before starting moving to GPU, you can simulate your GPU scan in this function first.
         */
        void scan(int n, int *odata, const int *idata) {
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
            timer().startCpuTimer();
            // TODO
            int count = 0;
            for (int i = 0; i < n; i++) {
                if (idata[i] != 0) {
                    odata[count] = idata[i];
                    count++;
                }
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
            int *bools = new int[n];
            int *indices = new int[n];
            timer().startCpuTimer();
            // TODO
            for (int i = 0; i < n; i++) {
                bools[i] = idata[i] != 0 ? 1 : 0;
            }
            exclusiveScan(n, indices, bools);
            for (int i = 0; i < n; i++) {
                if (bools[i] == 1) {
                    odata[indices[i]] = idata[i];
                }
            }
            int count = indices[n - 1] + bools[n - 1];
            timer().endCpuTimer();
            delete[] bools;
            delete[] indices;
            return count;
        }

        // std::sort, to check and compare the radix sort
        void sort(int n, int *odata, const int *idata) {
            std::copy(idata, idata + n, odata);
            timer().startCpuTimer();
            std::sort(odata, odata + n);
            timer().endCpuTimer();
        }
    }
}

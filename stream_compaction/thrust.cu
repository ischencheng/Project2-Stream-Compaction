#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/scan.h>
#include "common.h"
#include "thrust.h"
#include "device_util.cuh"

namespace StreamCompaction {
    namespace Thrust {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            Common::validateSize(n);
            thrust::device_vector<int> input(n), output(n);
            if (n) Common::cudaCheck(cudaMemcpy(thrust::raw_pointer_cast(input.data()), idata, size_t(n) * sizeof(int), cudaMemcpyHostToDevice), "Thrust input");
            timer().startGpuTimer();
            // TODO use `thrust::exclusive_scan`
            // example: for device_vectors dv_in and dv_out:
            // thrust::exclusive_scan(dv_in.begin(), dv_in.end(), dv_out.begin());
            if (n) thrust::exclusive_scan(input.begin(), input.end(), output.begin());
            timer().endGpuTimer();
            Common::cudaCheck(cudaGetLastError(), "Thrust scan");
            if (n) Common::cudaCheck(cudaMemcpy(odata, thrust::raw_pointer_cast(output.data()), size_t(n) * sizeof(int), cudaMemcpyDeviceToHost), "Thrust output");
        }
    }
}

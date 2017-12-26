#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
int main()
{
	int device_no;

	//get device number
	cudaGetDeviceCount(&device_no);

	//for each device find the props
	int i, driverVersion, runtimeVersion;
	for(i = 0; i < device_no; i++)
	{
		cudaDeviceProp properties;
		cudaGetDeviceProperties(&properties, i);
		printf("Name of device %d: %s\n", i, properties.name);

		cudaDriverGetVersion(&driverVersion);
		cudaRuntimeGetVersion(&runtimeVersion);
		printf("\tCUDA driver version: %d.%d\n", driverVersion/1000, (driverVersion%100)/10);
		printf("\tCUDA runtime Version: %d.%d\n", runtimeVersion/1000, (runtimeVersion%100)/10);
		printf("\tCUDA capability version number: %d.%d\n", properties.major, properties.minor);

		printf("\tMemory clock rate (KHz): %.0f Mhz\n", properties.memoryClockRate * 1e-3f);
		printf("\tMemory bus width (bits): %d\n", properties.memoryBusWidth);
		printf("\tPeak memory bandwidth: (GB/s): %f\n", 2.0*properties.memoryClockRate*(properties.memoryBusWidth/8)/1.0e6);
		printf("\tTotal constant memory (bytes): %lu\n", properties.totalGlobalMem);
		printf("\tTotal global memory: %.0f MBytes (%llu bytes)\n", (float)properties.totalGlobalMem/1048576.0f, (unsigned long long) properties.totalGlobalMem);
		printf("\tMaximum shared memory available on a thread block (bytes): %lu\n", properties.sharedMemPerBlock);
		printf("\tMaximum number of 32-bit registers on a thread block: %d\n", properties.regsPerBlock);
		printf("\tWarp size: %d\n", properties.warpSize);
		printf("\tMaximum number of threads per block: %d\n", properties.maxThreadsPerBlock);
		printf("\tMaximum size of each dimension of a block: %d, %d, %d\n", properties.maxThreadsDim[0], properties.maxThreadsDim[1], properties.maxThreadsDim[2]);
		printf("\tMaximum size of each dimension of a grid: %d, %d, %d\n", properties.maxGridSize[0], properties.maxGridSize[1], properties.maxGridSize[2]);
		printf("\tClock Rate (KHz): %d\n\n", properties.clockRate);
	}
}


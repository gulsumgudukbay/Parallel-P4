/**
 * Angle Between Two Vectors A and B
 *
 * Author: Gulsum Gudukbay
 * Date: 23 December 2017
 *
 */

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

// double precision atomic add function
__device__ double atomicAdd2(double* address, double val)
{
    unsigned long long int* address_as_ull = (unsigned long long int*)address;

    unsigned long long int old = *address_as_ull, assumed;

    do{ assumed = old;
		old = atomicCAS(address_as_ull, assumed,__double_as_longlong(val +__longlong_as_double(assumed)));
    } while (assumed != old);

    return __longlong_as_double(old);
}

__device__ void dot_product(const double *A, const double *B, int numElements, int blockSize, int width_thread, double *result)
{
	int start = blockIdx.x * blockSize + threadIdx.x;
	for(int i = start; i < start+width_thread; i++)
		result[blockIdx.x]+= A[start+i]*B[start+i];
		//atomicAdd2(&result[blockIdx.x], A[start + i] * B[start+i]);
}

__device__ void mag_squared(const double *A, int numElements, int blockSize, int width_thread, double *result)
{
	int start = blockIdx.x * blockSize + threadIdx.x;

	//sum all elements squared in the block
	for(int i = start; i < start+width_thread; i++)
	{
		printf("A[%d]: %.2f\n", start+i, A[start+i]);
		result[blockIdx.x] += pow(A[start+i], 2);
		//atomicAdd2(&result[blockIdx.x], pow(A[start + i], 2));
	}
}

__global__ void find_angle(const double *A, const double *B, int numElements, int blockSize, int width_thread, double *mag1, double *mag2, double *dot_prod)
{
	printf("\nhello\n");
	for(int i = 0; i < numElements; i++)
		printf("A[]: %.2f\n", i);
	mag_squared(A, numElements, blockSize, width_thread, mag1);
	mag_squared(B, numElements, blockSize, width_thread, mag2);
	dot_product(A, B, numElements, blockSize, width_thread, dot_prod);
	__syncthreads();
}



double findAngleCPU(const double *A, const double *B, int numElements)
{
	double res = 0.0;
	double dot_prod = 0.0;
	double mag1 = 0.0;
	double mag2 = 0.0;

	for(int i = 0; i < numElements; i++)
	{
		dot_prod += A[i] * B[i];
		mag1 += pow(A[i], 2);
		mag2 += pow(B[i], 2);
	}

	res = acos(dot_prod/(sqrt(mag1)*sqrt(mag2)));
	return res;
}

int main()
{
	int N, blockSize;
	double *A, *B, *d_A, *d_B;
	double *dot_prod, *mag1, *mag2;
	double *h_dot_prod, *h_mag1, *h_mag2;
	srand (time ( NULL));

	N = 1000000;
	blockSize = 512;
	cudaDeviceProp properties;
	cudaGetDeviceProperties(&properties, 0);
	printf("using %d multiprocessors\n", properties.multiProcessorCount);
	printf("max threads per processor: %d\n", properties.maxThreadsPerMultiProcessor);

	int width_thread = N / blockSize/512;
	int no_of_blocks = (int)ceil( N / blockSize/512);

	printf("Number of blocks will be created: %d\n", no_of_blocks);

	A = (double*)malloc(N * sizeof(double));
	B = (double*)malloc(N * sizeof(double));

	h_dot_prod = (double*)malloc(no_of_blocks * sizeof(double));
	h_mag1 = (double*)malloc(no_of_blocks * sizeof(double));
	h_mag2 = (double*)malloc(no_of_blocks * sizeof(double));

	double dot_product, magnitude1, magnitude2;
	dot_product = 0.0;
	magnitude1 = 0.0;
	magnitude2 = 0.0;

	//fill in the arrays with random numbers
	for(int i = 0; i < N; i++)
	{
		A[i] = rand() / (RAND_MAX / 1000);
	}
	for(int i = 0; i < N; i++)
	{
		B[i] = rand() / (RAND_MAX / 1000);
	}

	//Compute angle on CPU
	printf("Angle on CPU: %.2f\n", (180.0 / M_PI)*findAngleCPU(A, B, N));


	cudaMalloc(&d_A, N * sizeof(double));
	cudaMalloc(&d_B, N * sizeof(double));
	cudaMalloc(&dot_prod, no_of_blocks*sizeof(double));
	cudaMalloc(&mag1, no_of_blocks*sizeof(double));
	cudaMalloc(&mag2, no_of_blocks*sizeof(double));

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);

	cudaMemcpy(d_A, A, N*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, B, N*sizeof(double), cudaMemcpyHostToDevice);
	cudaEventRecord(stop);

	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("Time for Host to Device transfer: %f ms\n", milliseconds);
	find_angle<<<no_of_blocks, blockSize>>>(d_A, d_B, N, blockSize, width_thread, mag1, mag2, dot_prod);
	cudaError_t error = cudaGetLastError();
		if (error != cudaSuccess) {
		  fprintf(stderr, "ERROR: %s \n", cudaGetErrorString(error));
	}
	cudaThreadSynchronize();

	for(int i = 0; i < no_of_blocks; i++)
	{
		magnitude1 += mag1[i];
		magnitude2 += mag2[i];
		dot_product += dot_prod[i];
	}

	printf("magnitude1: %.2f, magnitude2: %.2f, dot_product: %.2f\n", magnitude1, magnitude2, dot_product);

	cudaMemcpy(h_dot_prod, dot_prod, no_of_blocks*sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_mag1, mag1, no_of_blocks*sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_mag2, mag2, no_of_blocks*sizeof(double), cudaMemcpyDeviceToHost);



	for(int i = 0; i < no_of_blocks; i++)
	{
		dot_product += h_dot_prod[i];
		magnitude1 += h_mag1[i];
		magnitude2 += h_mag2[i];
	}

	magnitude1 = sqrt(magnitude1);
	magnitude2 = sqrt(magnitude2);

	double result = acos(dot_product/(magnitude1*magnitude2));

	printf("Angle: %f\n", (180.0 / M_PI)*result);


	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(dot_prod);
	cudaFree(mag1);
	cudaFree(mag2);

}


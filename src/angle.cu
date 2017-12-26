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
// taken from https://devtalk.nvidia.com/default/topic/763119/atomic-add-operation/
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
	int start = width_thread*(blockIdx.x * blockSize + threadIdx.x);
	double sum = 0.0;

	for(int i = start; i < start+width_thread; i++)
	{
		if(i < numElements)
			sum += A[i] * B[i];
	}
	atomicAdd2(&result[blockIdx.x], sum);

}

__device__ void mag_squared(const double *A, int numElements, int blockSize, int width_thread, double *result)
{
	int start =width_thread*(blockIdx.x * blockSize + threadIdx.x);

	double sum = 0.0;
	//sum all elements squared in the block
	for(int i = start; i < start+width_thread; i++)
	{
		if(i < numElements)
			sum+= pow(A[i],2);
	}
	atomicAdd2(&result[blockIdx.x], sum);
}

__global__ void find_angle(const double *A, const double *B, int numElements, int blockSize, int width_thread, double *mag1, double *mag2, double *dot_prod)
{

	mag_squared(A, numElements, blockSize, width_thread+1, mag1);
	mag_squared(B, numElements, blockSize, width_thread+1, mag2);
	dot_product(A, B, numElements, blockSize, width_thread+1, dot_prod);
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

	mag1 = sqrt(mag1);
	mag2 = sqrt(mag2);

	res = acos(dot_prod/(mag1*mag2));
	return res;
}

int main(int argc, char *argv[])
{
	srand (58);

	/*FILE* in = fopen("input.txt", "w+");

	fprintf(in, "%f\n", (float)1000000);
	for( int i = 0; i < 2000000; i++)
	{
		fprintf(in, "%f\n", (float)(rand() / (RAND_MAX / 100)));
	}
	fclose(in);
*/

	int N, blockSize, threadElts;
	double *A, *B, *d_A, *d_B;
	double *dot_prod, *mag1, *mag2;
	double *h_dot_prod, *h_mag1, *h_mag2;

	char* filename;
	dot_prod = NULL;
	mag1 = NULL;
	mag2 = NULL;
	threadElts = 256;
	N = atoi(argv[1]);
	blockSize = atoi(argv[2]);

	if(argc == 4)
		filename = argv[3];


	cudaEvent_t start4, stop4;
	cudaEventCreate(&start4);
	cudaEventCreate(&stop4);
	cudaEventRecord(start4);

	if(argc == 3)
	{
		A = (double*)malloc(N * sizeof(double));
		B = (double*)malloc(N * sizeof(double));

		//fill in the arrays with random numbers
		for(int i = 0; i < N; i++)
		{
			A[i] = rand() / (RAND_MAX / 100);
		}
		for(int i = 0; i < N; i++)
		{
			B[i] = rand() / (RAND_MAX / 100);
		}
	}
	else
	{
		FILE * file;
		int i;
		float tmp;

		if ((file = fopen(filename, "r+")) == NULL)
		{
			printf("ERROR: file open failed\n");
			return -1;
		}
		fscanf(file,"%f", &tmp);
		N = (int)tmp;
		printf("%f\n", tmp);
		A = (double*)malloc(N * sizeof(double));
		B = (double*)malloc(N * sizeof(double));

		for(i = 0; i < N; i++)
		{
			fscanf(file,"%f", &tmp);
			A[i] = tmp;
		}
		for(i = 0; i < N; i++)
		{
			fscanf(file,"%f", &tmp);
			B[i] = tmp;
		}
		fclose(file);

	}

	cudaEventRecord(stop4);

	cudaEventSynchronize(stop4);
	float milliseconds4 = 0;
	cudaEventElapsedTime(&milliseconds4, start4, stop4);
	printf("Time for the array generation: %f ms\n", milliseconds4);

	int no_of_blocks = (int)ceil( N / blockSize / threadElts)+1;

	printf("\nInfo\n______________________________________________________\n");
	printf("Number of elements: %d\n", N);
	printf("Number of threads per block: %d\n", blockSize);
	printf("Number of blocks will be created: %d\n", no_of_blocks);
	printf("\nTime\n______________________________________________________\n");


	h_dot_prod = (double*)malloc(no_of_blocks * sizeof(double));
	h_mag1 = (double*)malloc(no_of_blocks * sizeof(double));
	h_mag2 = (double*)malloc(no_of_blocks * sizeof(double));

	double dot_product, magnitude1, magnitude2;
	dot_product = 0.0;
	magnitude1 = 0.0;
	magnitude2 = 0.0;

	//Compute angle on CPU
	cudaEvent_t start3, stop3;
	cudaEventCreate(&start3);
	cudaEventCreate(&stop3);
	cudaEventRecord(start3);
	float cpu_result = (float)((180.0 / M_PI)*findAngleCPU(A, B, N));
	cudaEventRecord(stop3);

	cudaEventSynchronize(stop3);
	float milliseconds3 = 0;
	cudaEventElapsedTime(&milliseconds3, start3, stop3);
	printf("Time for the CPU function: %f ms\n", milliseconds3);

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

	//KERNEL
	cudaEvent_t start2, stop2;
	float milliseconds2 = 0;

	cudaEventCreate(&start2);
	cudaEventCreate(&stop2);
	cudaEventRecord(start2);
	find_angle<<<no_of_blocks, blockSize>>>(d_A, d_B, N, blockSize, threadElts, mag1, mag2, dot_prod);
	cudaDeviceSynchronize();
	cudaEventRecord(stop2);

	cudaEventSynchronize(stop2);
	cudaEventElapsedTime(&milliseconds2, start2, stop2);
	printf("Time for the kernel execution: %f ms\n", milliseconds2);

	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess) {
	  fprintf(stderr, "ERROR: %s \n", cudaGetErrorString(error));
	}

	cudaEvent_t start5, stop5;
	float milliseconds5 = 0;

	cudaEventCreate(&start5);
	cudaEventCreate(&stop5);
	cudaEventRecord(start5);

	cudaMemcpy(h_dot_prod, dot_prod, no_of_blocks*sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_mag1, mag1, no_of_blocks*sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_mag2, mag2, no_of_blocks*sizeof(double), cudaMemcpyDeviceToHost);

	cudaEventRecord(stop5);

	cudaEventSynchronize(stop5);
	cudaEventElapsedTime(&milliseconds5, start5, stop5);
	printf("Time for the Device to Host transfer: %f ms\n", milliseconds5);
	printf("Total execution time for GPU: %f ms\n", milliseconds5 + milliseconds2 + milliseconds);

	for(int i = 0; i < no_of_blocks; i++)
	{
		magnitude1 += h_mag1[i];
		magnitude2 += h_mag2[i];
		dot_product += h_dot_prod[i];
	}

	magnitude1 = sqrt(magnitude1);
	magnitude2 = sqrt(magnitude2);

	//printf("magnitude1: %.2f, magnitude2: %.2f, dot_product: %.2f\n", (float)magnitude1, (float)magnitude2, (float)dot_product);

	double result = acos(dot_product/(magnitude1*magnitude2));
	printf("\nResults\n____________________________________________________\n");
	printf("CPU result: %f\n", cpu_result);
	printf("GPU result: %f\n\n", (float)((180.0 / M_PI)*result));


	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(dot_prod);
	cudaFree(mag1);
	cudaFree(mag2);

	return 0;
}


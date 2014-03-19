#include "cuda_runtime.h"
#include "device_launch_parameters.h"
//#include <stdio.h>

//device functions
__device__ int getGlobalIdx_1D_1D()
{
	return blockIdx.x *blockDim.x + threadIdx.x;
}

__device__ int getGlobalIdx_1D_2D()
{
	return blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
}

__device__ int getGlobalIdx_1D_3D()
{
	return blockIdx.x * blockDim.x * blockDim.y * blockDim.z + threadIdx.z * blockDim.y * blockDim.x + threadIdx.y * blockDim.x + threadIdx.x;
}

__device__ int getGlobalIdx_2D_1D()
{
	int blockId = blockIdx.y * gridDim.x 
				+ blockIdx.x;			 	

	int threadId = blockId * blockDim.x + threadIdx.x; 

	return threadId;
}

__device__ int getGlobalIdx_2D_2D()
{
	int blockId = blockIdx.x 
				+ blockIdx.y * gridDim.x; 

	int threadId = blockId * (blockDim.x * blockDim.y)
				 + (threadIdx.y * blockDim.x)
				 + threadIdx.x;

	return threadId;
}

__device__ int getGlobalIdx_2D_3D()
{
	int blockId = blockIdx.x 
				+ blockIdx.y * gridDim.x; 

	int threadId = blockId * (blockDim.x * blockDim.y * blockDim.z)
				 + (threadIdx.z * (blockDim.x * blockDim.y))
				 + (threadIdx.y * blockDim.x)
				 + threadIdx.x;

	return threadId;
}

__device__ int getGlobalIdx_3D_1D()
{
	int blockId = blockIdx.x 
				+ blockIdx.y * gridDim.x 
				+ gridDim.x * gridDim.y * blockIdx.z; 

	int threadId = blockId * blockDim.x + threadIdx.x;

	return threadId;
}

__device__ int getGlobalIdx_3D_2D()
{
	int blockId = blockIdx.x 
				+ blockIdx.y * gridDim.x 
				+ gridDim.x * gridDim.y * blockIdx.z; 

	int threadId = blockId * (blockDim.x * blockDim.y)
				 + (threadIdx.y * blockDim.x)
				 + threadIdx.x;

	return threadId;
}

__device__ int getGlobalIdx_3D_3D()
{
	int blockId = blockIdx.x 
				+ blockIdx.y * gridDim.x 
				+ gridDim.x * gridDim.y * blockIdx.z; 

	int threadId = blockId * (blockDim.x * blockDim.y * blockDim.z)
				 + (threadIdx.z * (blockDim.x * blockDim.y))
				 + (threadIdx.y * blockDim.x)
				 + threadIdx.x;

	return threadId;
}
/*
//kernels
__global__ void kernel_1D_1D()
{
	printf("Local thread ID: %i   Global thread ID: %i\n", threadIdx.x, getGlobalIdx_1D_1D());
}

__global__ void kernel_1D_2D()
{
	printf("Local thread IDs: (%i,%i)   Global thread ID: %i\n", threadIdx.x, threadIdx.y, getGlobalIdx_1D_2D());
}

__global__ void kernel_1D_3D()
{
	printf("Local thread IDs: (%i,%i,%i)   Global thread ID: %i\n", threadIdx.x, threadIdx.y, threadIdx.z, getGlobalIdx_1D_3D());
}

__global__ void kernel_2D_1D()
{
	printf("Local thread ID: %i   Global thread ID: %i\n", threadIdx.x, getGlobalIdx_2D_1D());
}

__global__ void kernel_2D_2D()
{
	printf("Local thread IDs: (%i,%i)   Global thread ID: %i\n", threadIdx.x, threadIdx.y, getGlobalIdx_2D_2D());
}

__global__ void kernel_2D_3D()
{
	printf("Local thread IDs: (%i,%i,%i)   Global thread ID: %i\n", threadIdx.x, threadIdx.y, threadIdx.z, getGlobalIdx_2D_3D());
}

__global__ void kernel_3D_1D()
{
	printf("Local thread ID: %i   Global thread ID: %i\n", threadIdx.x, getGlobalIdx_3D_1D());
}

__global__ void kernel_3D_2D()
{
	printf("Local thread IDs: (%i,%i)   Global thread ID: %i\n", threadIdx.x, threadIdx.y, getGlobalIdx_3D_2D());
}

__global__ void kernel_3D_3D()
{
	printf("Local thread IDs: (%i,%i,%i)   Global thread ID: %i\n", threadIdx.x, threadIdx.y, threadIdx.z, getGlobalIdx_3D_3D());
}

int main()
{
	printf("\nLaunching kernel as 1D grid of 1D blocks...\n");
	kernel_1D_1D<<<dim3(2,1,1), dim3(2,1,1)>>>();
	cudaDeviceReset();

	printf("\nLaunching kernel as 1D grid of 2D blocks...\n");
	kernel_1D_2D<<<dim3(2,1,1), dim3(2,2,1)>>>();
	cudaDeviceReset();

	printf("\nLaunching kernel as 1D grid of 3D blocks...\n");
	kernel_1D_3D<<<dim3(2,1,1), dim3(2,2,2)>>>();
	cudaDeviceReset();


	printf("\nLaunching kernel as 2D grid of 1D blocks...\n");
	kernel_2D_1D<<<dim3(2,2,1), dim3(2,1,1)>>>();
	cudaDeviceReset();

	printf("\nLaunching kernel as 2D grid of 2D blocks...\n");
	kernel_2D_2D<<<dim3(2,2,1), dim3(2,2,1)>>>();
	cudaDeviceReset();

	printf("\nLaunching kernel as 2D grid of 3D blocks...\n");
	kernel_2D_3D<<<dim3(2,2,1), dim3(2,2,2)>>>();
	cudaDeviceReset();


	printf("\nLaunching kernel as 3D grid of 1D blocks...\n");
	kernel_3D_1D<<<dim3(2,2,2), dim3(2,1,1)>>>();
	cudaDeviceReset();

	printf("\nLaunching kernel as 3D grid of 2D blocks...\n");
	kernel_3D_2D<<<dim3(2,2,2), dim3(2,2,1)>>>();
	cudaDeviceReset();

	printf("\nLaunching kernel as 3D grid of 3D blocks...\n");
	kernel_3D_3D<<<dim3(2,2,2), dim3(2,2,2)>>>();
	cudaDeviceReset();

	return 0;
}
*/

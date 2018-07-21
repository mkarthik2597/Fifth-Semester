#include<stdio.h>
int main()
{
	cudaDeviceProp properties;
	cudaGetDeviceProperties(&properties,0);  //Only one device attached (Can be found by cudaGetDeviceCount())
	
	printf("Device name:%s\n",properties.name);
	printf("Total global memory:%ld bytes\n",properties.totalGlobalMem);
	printf("Total constant memory:%ld bytes\n",properties.totalConstMem);
	printf("Warp size:%d\n",properties.warpSize);
	printf("Maximum threads per block:%ld\n",properties.maxThreadsPerBlock);
	printf("Maximum block dimension:(%ld,%ld,%ld)\n",properties.maxThreadsDim[0],properties.maxThreadsDim[1],properties.maxThreadsDim[2]);
	printf("Maximum grid dimensions:(%ld,%ld,%ld)\n",properties.maxGridSize[0],properties.maxGridSize[1],properties.maxGridSize[2]);	
	printf("Shared memory per block:%ld bytes\n",properties.sharedMemPerBlock);
	printf("Compute capability:%d.%d\n",properties.major,properties.minor);
}
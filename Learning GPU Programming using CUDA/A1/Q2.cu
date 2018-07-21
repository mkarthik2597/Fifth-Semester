#include<stdio.h>
#include<stdlib.h>

/*Kernel functions must be declared before defining them*/
__global__ void ArrayAddKernel(float*,float*,float*);

int main()
{
	/*Choose n value between 10 and 1024 only. Go through the code to see why*/
	int n=10;
	
	/*Dynamically allocate memory for the arr1, arr2 and sum array on the host memory*/
	float* arr1=(float*)malloc(n*sizeof(float));
	float* arr2=(float*)malloc(n*sizeof(float));
	float* sum=(float*)malloc(n*sizeof(float));
	
	srand(time(NULL));
	
	/*Populate the arrays with random numbers*/
	for(int i=0;i<n;i++)
	{
		arr1[i]=rand()%5;
		arr2[i]=rand()%100;
	}
	
	/*Total no.of bytes in each array*/
	int size=n*sizeof(float);
	
	/*Pointers to memory locations on the device*/
	float *arr1_d, *arr2_d, *sum_d;
	
	/*Allocate memory on device*/
	cudaMalloc((void**)&arr1_d,size);
	cudaMalloc((void**)&arr2_d,size);
	cudaMalloc((void**)&sum_d,size);
	
	/*Copy arr1,arr2 from host memory to device memory*/
	cudaMemcpy(arr1_d,arr1,size,cudaMemcpyHostToDevice);
	cudaMemcpy(arr2_d,arr2,size,cudaMemcpyHostToDevice);
	
	/*Each block has a dimension (n,1,1). n<=1024*/
	dim3 dimBlock(n);
	/*The grid has only one block*/
	dim3 dimGrid(1);
	/*Kernel launch*/
	ArrayAddKernel<<<dimGrid,dimBlock>>>(arr1_d,arr2_d,sum_d);
	
	/*Transfer the calculated sum array from device memory to host memory*/
	cudaMemcpy(sum,sum_d,size,cudaMemcpyDeviceToHost);
	
	/*Display the first 10 elements of arr1,arr2,sum. n>=10*/
	for(int i=0;i<10;i++)
	printf("%f %f  %f\n",arr1[i],arr2[i],sum[i]);
    
	printf("\n");
	
	cudaFree(arr1_d);
	cudaFree(arr2_d);
	cudaFree(sum_d);	
}

__global__ void ArrayAddKernel(float* arr1_d, float* arr2_d, float* sum_d)
{
	sum_d[threadIdx.x]=arr1_d[threadIdx.x]+arr2_d[threadIdx.x];
}
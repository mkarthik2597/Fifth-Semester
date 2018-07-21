#include<stdio.h>
#include<stdlib.h>
#include<math.h>
__global__ void ArrayAddKernel(float*,float*,float*,int);

int main()
{
	int n=16500;
	float* arr1=(float*)malloc(n*sizeof(float));
	float* arr2=(float*)malloc(n*sizeof(float));
	float* sum=(float*)malloc(n*sizeof(float));
	
	srand(time(NULL));
	
	for(int i=0;i<n;i++)
	{
		arr1[i]=rand()%5;
		arr2[i]=rand()%100;
	}
	
	int size=n*sizeof(float);
	
	float *arr1_d, *arr2_d, *sum_d;
	
	cudaMalloc((void**)&arr1_d,size);
	cudaMalloc((void**)&arr2_d,size);
	cudaMalloc((void**)&sum_d,size);
	
	cudaMemcpy(arr1_d,arr1,size,cudaMemcpyHostToDevice);
	cudaMemcpy(arr2_d,arr2,size,cudaMemcpyHostToDevice);
	
	/*Each block of thread has dimensions (1024,1,1);*/
	dim3 dimBlock(1024);
	dim3 dimGrid(ceil(n/1024.0));
	ArrayAddKernel<<<dimGrid,dimBlock>>>(arr1_d,arr2_d,sum_d,n);
	
	cudaMemcpy(sum,sum_d,size,cudaMemcpyDeviceToHost);
	
	for(int i=0;i<10;i++)
	printf("%f %f  %f\n",arr1[i],arr2[i],sum[i]);
    
	printf("\n");
	
	cudaFree(arr1_d);
	cudaFree(arr2_d);
	cudaFree(sum_d);	
}

__global__ void ArrayAddKernel(float* arr1_d, float* arr2_d, float* sum_d,int n)
{
	int index=blockIdx.x*blockDim.x+threadIdx.x;
	if(index<n)
	sum_d[index]=arr1_d[index]+arr2_d[index];
}

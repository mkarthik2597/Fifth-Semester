#include<stdio.h>
#include<stdlib.h>
#include<math.h>

#define WARP_SIZE 32

__global__ void scan(int*);
void printArray(int*);

int main()
{
	int* input=new int[WARP_SIZE];
	for(int i=0;i<WARP_SIZE;i++)
	input[i]=rand()%100;
	
	int* input_d;
	
	cudaMalloc((void**)&input_d,WARP_SIZE*sizeof(int));
	cudaMemcpy(input_d,input,WARP_SIZE*sizeof(int),cudaMemcpyHostToDevice);

	int Solution[WARP_SIZE];
	Solution[0]=input[0];
	
	for(int i=1;i<WARP_SIZE;i++)
	Solution[i]=input[i]+Solution[i-1];
		
	scan<<<1,WARP_SIZE,WARP_SIZE*sizeof(int)>>>(input_d);
	
	cudaMemcpy(input,input_d,WARP_SIZE*sizeof(int),cudaMemcpyDeviceToHost);
	
	int i;
	for(i=0;i<WARP_SIZE;i++)
	if(Solution[i]!=input[i])
	{
		printf("Wrong answer\n");
		printf("Disparity at %d\n",i);
		break;
	}
	if(i==WARP_SIZE)
	printf("Solution is right!\n");
	
}

__global__ void scan(int* input_d)
{
	int tx=threadIdx.x;
	
	extern __shared__ int partialSum[];
	partialSum[tx]=input_d[tx];
	__syncthreads();
	
	for(int s=1;s<blockDim.x;s<<=1)
	{
		if(tx>=s)
		partialSum[tx]+=partialSum[tx-s];
	}
	
	input_d[tx]=partialSum[tx];
}

void printArray(int* arr)
{
	for(int i=0;i<WARP_SIZE;i++)
	printf("%d ",arr[i]);

	printf("\n");
}
#include<stdio.h>
#include<stdlib.h>
#include<math.h>

#define BLOCK_SIZE 1024

__global__ void scan(int*);
void printArray(int*);

int main()
{
	int* input=new int[BLOCK_SIZE];
	for(int i=0;i<BLOCK_SIZE;i++)
	input[i]=rand()%100;
	
	int* input_d;
	
	cudaMalloc((void**)&input_d,BLOCK_SIZE*sizeof(int));
	cudaMemcpy(input_d,input,BLOCK_SIZE*sizeof(int),cudaMemcpyHostToDevice);

	int Solution[BLOCK_SIZE];
	Solution[0]=input[0];
	
	for(int i=1;i<BLOCK_SIZE;i++)
	Solution[i]=input[i]+Solution[i-1];
		
	scan<<<1,BLOCK_SIZE,2*BLOCK_SIZE*sizeof(int)>>>(input_d);
	
	cudaMemcpy(input,input_d,BLOCK_SIZE*sizeof(int),cudaMemcpyDeviceToHost);
	
	int i;
	for(i=0;i<BLOCK_SIZE;i++)
	if(Solution[i]!=input[i])
	{
		printf("Wrong answer\n");
		printf("Disparity at %d\n",i);
		break;
	}
	if(i==BLOCK_SIZE)
	printf("Solution is right!\n");
}

__global__ void scan(int* input_d)
{
	int tx=threadIdx.x;
	int outputRow=0,inputRow=1;
	
	extern __shared__ int partialSum[];
	partialSum[inputRow*BLOCK_SIZE+tx]=input_d[tx];
	__syncthreads();
	
	for(int s=1;s<blockDim.x;s<<=1)
	{
		if(tx>=s)
		partialSum[outputRow*BLOCK_SIZE+tx]=partialSum[inputRow*BLOCK_SIZE+tx]+partialSum[inputRow*BLOCK_SIZE+tx-s];
		else
		partialSum[outputRow*BLOCK_SIZE+tx]=partialSum[inputRow*BLOCK_SIZE+tx];
	
		__syncthreads();
		
		outputRow=1-outputRow;
		inputRow=1-inputRow;
	}
	
	input_d[tx]=partialSum[inputRow*BLOCK_SIZE+tx];
}

void printArray(int* arr)
{
	for(int i=0;i<BLOCK_SIZE;i++)
	printf("%d ",arr[i]);

	printf("\n");
}
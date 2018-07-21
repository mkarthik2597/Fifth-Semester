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
	input[i]=rand()%10;
	
	int* input_d;
	
	cudaMalloc((void**)&input_d,BLOCK_SIZE*sizeof(int));
	cudaMemcpy(input_d,input,BLOCK_SIZE*sizeof(int),cudaMemcpyHostToDevice);

	/* Scan is exclusive */
	int Solution[BLOCK_SIZE];
	Solution[0]=0;
	
	for(int i=1;i<BLOCK_SIZE;i++)
	Solution[i]=input[i-1]+Solution[i-1];

	scan<<<1,BLOCK_SIZE,BLOCK_SIZE*sizeof(int)>>>(input_d);
	
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
	
	extern __shared__ int partialSum[];
	partialSum[tx]=input_d[tx];
	__syncthreads();
	
	/* Upsweep phase (Reduction)*/
	for(int s=1;s<blockDim.x;s<<=1)
	{
		if((tx+1)%(2*s)==0)
		partialSum[tx]+=partialSum[tx-s];
	
		__syncthreads();
	}
	
	if(tx==BLOCK_SIZE-1)
	partialSum[tx]=0;
	
	/* Downsweep phase*/
	int temp;
	for(int s=blockDim.x/2;s>0;s>>=1)
	{
		if((tx+1)%(2*s)==0)
		{
			temp=partialSum[tx];
			partialSum[tx]+=partialSum[tx-s];
			partialSum[tx-s]=temp;
		}
		
		__syncthreads();
	}
	
	input_d[tx]=partialSum[tx];
}

void printArray(int* arr)
{
	for(int i=0;i<BLOCK_SIZE;i++)
	printf("%d ",arr[i]);

	printf("\n");
}
#include<stdio.h>
#include<stdlib.h>
#include<math.h>

#define SIZE int(pow(2,20))
#define BLOCK_SIZE 1024
#define NUM_BLOCKS SIZE/BLOCK_SIZE

__global__ void blockScan(int*,int*);
__global__ void blocksumScan(int*,int*);
void printArray(int*);

int main()
{
	int* input=new int[SIZE];
	for(int i=0;i<SIZE;i++)
	input[i]=rand()%10;
	
	int* auxilliary=new int[NUM_BLOCKS];
	
	int* input_d,*auxilliary_d;
	
	cudaMalloc((void**)&input_d,SIZE*sizeof(int));
	cudaMemcpy(input_d,input,SIZE*sizeof(int),cudaMemcpyHostToDevice);
	
	cudaMalloc((void**)&auxilliary_d,(NUM_BLOCKS)*sizeof(int));

	int Solution[SIZE];
	Solution[0]=input[0];
	
	for(int i=1;i<SIZE;i++)
	Solution[i]=input[i]+Solution[i-1];

	blockScan<<<NUM_BLOCKS,BLOCK_SIZE,BLOCK_SIZE*sizeof(int)>>>(input_d,auxilliary_d);
	blocksumScan<<<1,NUM_BLOCKS,NUM_BLOCKS*sizeof(int)+1>>>(input_d,auxilliary_d);
	
	cudaMemcpy(input,input_d,SIZE*sizeof(int),cudaMemcpyDeviceToHost);

	int i;
	for(i=0;i<SIZE;i++)
	if(Solution[i]!=input[i])
	{
		printf("Wrong answer\n");
		printf("Disparity at %d\n",i);
		break;
	}
	if(i==SIZE)
	printf("Solution is right!\n");
}

__global__ void blockScan(int* input_d, int* auxilliary_d)
{
	int tx=threadIdx.x,bx=blockIdx.x;
	
	extern __shared__ int partialSum[];
	partialSum[tx]=input_d[bx*BLOCK_SIZE+tx];
	__syncthreads();
	
	/* Upsweep phase*/
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
	
	if(tx!=BLOCK_SIZE-1)
	input_d[bx*BLOCK_SIZE+tx]=partialSum[tx+1];
	else
	input_d[bx*BLOCK_SIZE+tx]=partialSum[tx]+input_d[bx*BLOCK_SIZE+tx];
	
	if(tx==BLOCK_SIZE-1)
	auxilliary_d[bx]=input_d[bx*BLOCK_SIZE+tx];
}

__global__ void blocksumScan(int* input_d, int* auxilliary_d)
{
	int tx=threadIdx.x;
	
	extern __shared__ int partialSum[];
	partialSum[tx]=auxilliary_d[tx];
	__syncthreads();
	
	/* Upsweep phase*/
	for(int s=1;s<blockDim.x;s<<=1)
	{
		if((tx+1)%(2*s)==0)
		partialSum[tx]+=partialSum[tx-s];
	
		__syncthreads();
	}
	
	if(tx==blockDim.x-1)
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

	if(tx!=0)
	{
		for(int i=0;i<BLOCK_SIZE;i++)
		input_d[(tx)*BLOCK_SIZE+i]+=partialSum[tx];
	}
}

void printArray(int* arr)
{
	for(int i=0;i<SIZE;i++)
	printf("%d ",arr[i]);

	printf("\n");
}

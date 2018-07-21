#include<stdio.h>
#include<stdlib.h>
#include<math.h>

#define size int(pow(2,25))

__global__ void reduction(int*);
void printArray(int*);

int main()
{
	int* input=new int[size];
	for(int i=0;i<size;i++)
	input[i]=rand()%100;

	int* input_d;

	cudaMalloc((void**)&input_d,size*sizeof(int));
	cudaMemcpy(input_d,input,size*sizeof(int),cudaMemcpyHostToDevice);


	dim3 dimBlock(1024);
	dim3 dimGrid(ceil(size/1024.0));


	/* Each element in the Solution array contains the solution for reduction of a block */
	int Solution[size/1024];

	int temp;
	for(int j=0;j<size/1024.0;j++)
	{
		temp=0;
		for(int i=0;i<dimBlock.x;i++)
		temp+=input[j*dimBlock.x+i];

		Solution[j]=temp;
	}

	reduction<<<dimGrid,dimBlock,dimBlock.x*sizeof(int)>>>(input_d);

	cudaMemcpy(input,input_d,size*sizeof(int),cudaMemcpyDeviceToHost);


	for(int j=0;j<size/1024.0;j++)
	{
		if(input[j*dimBlock.x]!=Solution[j])
		printf("Disparity at block %d\n",j);
	}
}

__global__ void reduction(int* input_d)
{
	int tx=threadIdx.x,bx=blockIdx.x;
	int inx=bx*blockDim.x+tx;

	extern __shared__ int partialSum[];
	partialSum[tx]=input_d[inx];
	__syncthreads();

	/* Strided index and Non-divergent branching */
	for(int s=blockDim.x/2;s>0;s>>=1)
	{
		if(tx<s)
		partialSum[tx]+=partialSum[tx+s];

		__syncthreads();
	}

	if(tx==0)
	input_d[inx]=partialSum[0];
}

void printArray(int* arr)
{
	for(int i=0;i<size;i++)
	printf("%d ",arr[i]);

	printf("\n");
}

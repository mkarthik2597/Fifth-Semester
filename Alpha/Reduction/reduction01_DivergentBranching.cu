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

	/* Calcualte Solution to verify with kernel answer */
	int Solution=0;

	for(int i=0;i<size;i++)
	Solution+=input[i];

	dim3 dimBlock(1024);
	dim3 dimGrid(ceil(size/1024.0));

	reduction<<<dimGrid,dimBlock,dimBlock.x*sizeof(int)>>>(input_d);

	cudaMemcpy(input,input_d,size*sizeof(int),cudaMemcpyDeviceToHost);

	int Answer=0;
	for(int i=0;i<size;i+=dimBlock.x)
	Answer+=input[i];

	if(Solution==Answer)
	printf("Solution is right\n");
	else
	{
		printf("Solution is wrong\n");
		printf("Answer:%d\n",Answer);
	}

}

__global__ void reduction(int* input_d)
{
	/* Has the least performance because of control divergence */
	int tx=threadIdx.x,bx=blockIdx.x;
	int inx=bx*blockDim.x+tx;

	extern __shared__ int partialSum[];
	partialSum[tx]=input_d[inx];
	__syncthreads();

	for(int s=1;s<blockDim.x;s<<=1)
	{
		if(tx%(2*s)==0)
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

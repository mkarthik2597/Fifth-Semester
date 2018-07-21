#include<stdio.h>
#include<stdlib.h>
#include<math.h>

#define SIZE int(pow(2,10))
#define RANGE int(pow(2,8))  /* Numbers are generated from 0 to RANGE-1*/
#define BLOCKSIZE 1024
#define NUMBLOCKS SIZE/BLOCKSIZE
#define nBITS int(log(RANGE)/log(2))	/* log(n)+1 bits to represent n */
#define digit(n,exp) (n/exp)%2

__global__ void RadixSort(int*,int,int*);
void CheckSolution(int*);

int main()
{
	int in=0,out=1;

	int** array=new int*[2];
	for(int i=0;i<2;i++)
	array[i]=new int[SIZE];

	for(int i=0;i<SIZE;i++)
	array[in][i]=rand()%RANGE;

	int* array_d;
	cudaMalloc((void**)&array_d,SIZE*sizeof(int));

	int host_histo[2]={};
	int* device_histo;
	cudaMalloc((void**)&device_histo,2*sizeof(int));

	int exp,rank;
	for(int i=0;i<nBITS;i++)
	{
		cudaMemcpy(array_d,array[in],SIZE*sizeof(int),cudaMemcpyHostToDevice);

		exp=pow(2,i);
		RadixSort<<<NUMBLOCKS,BLOCKSIZE>>>(array_d,exp,device_histo);

		cudaMemcpy(host_histo,device_histo,2*sizeof(int),cudaMemcpyDeviceToHost);
		/* The scan part has been moved outside because it should occur at a globally synchronised point */
		host_histo[1]+=host_histo[0];

		for(int j=SIZE-1;j>=0;j--)
		{
			rank=host_histo[digit(array[in][j],exp)]-1;

			array[out][rank]=array[in][j];

			host_histo[digit(array[in][j],exp)]--;
		}

		in=1-in;
		out=1-out;
	}

	CheckSolution(array[in]);
}

__global__ void RadixSort(int* array_d, int exp, int* device_histo)
{
	/* Histogram Calculation*/
	int tx=threadIdx.x,bx=blockIdx.x;
	int inx=bx*blockDim.x+tx;
	if(inx==0)
	{
		device_histo[0]=0;
		device_histo[1]=0;
	}

	__shared__ int shared_histo[2];
	if(tx==0)
	{
		shared_histo[0]=0;
		shared_histo[1]=0;
	}
	__syncthreads();

	atomicAdd(&shared_histo[digit(array_d[inx],exp)],1);
	__syncthreads();

	if(tx==0)
	{
		atomicAdd(&device_histo[0],shared_histo[0]);
		atomicAdd(&device_histo[1],shared_histo[1]);
	}
}

void CheckSolution(int* array)
{
	int i;
	for(i=0;i<SIZE-1;i++)
	if(array[i]>array[i+1])
	{
		printf("Solution is wrong!\n");
		break;
	}

	if(i==SIZE-1)
	printf("Solution is right!\n");
}

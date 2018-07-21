#include<stdio.h>
#include<stdlib.h>
#include<math.h>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/scan.h>
#include <thrust/device_ptr.h>

#define SIZE int(pow(2,10))
#define RANGE int(pow(2,5))  /* Numbers are generated from 0 to RANGE-1*/
#define BLOCKSIZE 1024
#define NUMBLOCKS SIZE/BLOCKSIZE
#define nBITS int(log(RANGE)/log(2))	/* log(n)+1 bits to represent n */
#define OFFSET 2						/* Number of bits for sorting in each pass */
#define HISTO_SIZE 4
#define digit(n,exp) (n/exp)%HISTO_SIZE


__global__ void RadixSort(int*,int,int*,int*);
__global__ void BlockWiseSort(int*,int,int*,int*);
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

	int host_histo[HISTO_SIZE];
	int* device_histo;
	cudaMalloc((void**)&device_histo,HISTO_SIZE*sizeof(int));

	int BlockWiseHistograms[NUMBLOCKS*HISTO_SIZE];
	int* BlockWiseHistograms_d;
	cudaMalloc((void**)&BlockWiseHistograms_d,NUMBLOCKS*HISTO_SIZE*sizeof(int));

	int exp,rank;
	for(int i=0;i<nBITS;i+=OFFSET)
	{
		cudaMemcpy(array_d,array[in],SIZE*sizeof(int),cudaMemcpyHostToDevice);

		exp=pow(2,i);

		/* Perform block wise histogram computation, add them up to the global histogram */
		RadixSort<<<NUMBLOCKS,BLOCKSIZE,HISTO_SIZE>>>(array_d,exp,device_histo,BlockWiseHistograms_d);

		cudaMemcpy(BlockWiseHistograms,BlockWiseHistograms_d,NUMBLOCKS*HISTO_SIZE*sizeof(int),cudaMemcpyDeviceToHost);

		/* Scan each of the block wise histograms */

		thrust::device_vector<int> ThrustBlockWiseHistograms(BlockWiseHistograms,BlockWiseHistograms+NUMBLOCKS*HISTO_SIZE);

		for(int j=0;j<NUMBLOCKS;j++)
		thrust::inclusive_scan(ThrustBlockWiseHistograms.begin()+j*BLOCKSIZE,
							   ThrustBlockWiseHistograms.begin()+(j+1)*BLOCKSIZE,
							   ThrustBlockWiseHistograms.begin()+j*BLOCKSIZE
							  );

		thrust::copy(ThrustBlockWiseHistograms.begin(), ThrustBlockWiseHistograms.end(), thrust::device_pointer_cast(BlockWiseHistograms));

		cudaMemcpy(BlockWiseHistograms_d,BlockWiseHistograms,NUMBLOCKS*HISTO_SIZE*sizeof(int),cudaMemcpyHostToDevice);

		int* BlockWiseSortedArray_d;
		cudaMalloc((void**)&BlockWiseSortedArray_d,SIZE*sizeof(int));

		/* Sort the array blockwise based on the scanned histograms */
		BlockWiseSort<<<NUMBLOCKS,BLOCKSIZE,HISTO_SIZE>>>(array_d,exp,BlockWiseSortedArray_d,BlockWiseHistograms_d);

		cudaMemcpy(host_histo,device_histo,HISTO_SIZE*sizeof(int),cudaMemcpyDeviceToHost);
		thrust::device_vector<int> GlobalHistogram(host_histo,host_histo+HISTO_SIZE);
		thrust::inclusive_scan(GlobalHistogram.begin(),GlobalHistogram.end(),GlobalHistogram.begin());
		thrust::copy(GlobalHistogram.begin(), GlobalHistogram.end(), thrust::device_pointer_cast(host_histo));

		/* Sort the array using the global histogram */
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

__global__ void RadixSort(int* array_d, int exp, int* device_histo,int* BlockWiseHistograms_d)
{
	int tx=threadIdx.x,bx=blockIdx.x;
	int inx=bx*blockDim.x+tx;

	extern __shared__ int shared_histo[];

	if(inx==0)
	{
		for(int i=0;i<HISTO_SIZE;i++)
		device_histo[i]=0;
	}

	if(tx==0)
	{
		for(int i=0;i<HISTO_SIZE;i++)
		shared_histo[i]=0;
	}
	__syncthreads();

	atomicAdd(&shared_histo[digit(array_d[inx],exp)],1);
	__syncthreads();

	if(tx==0)
	{
		for(int i=0;i<HISTO_SIZE;i++)
		{
			atomicAdd(&device_histo[i],shared_histo[i]);
			BlockWiseHistograms_d[bx*HISTO_SIZE+i]=shared_histo[i];
		}
	}
}

__global__ void BlockWiseSort(int* array_d,int exp,int* BlockWiseSortedArray_d, int* BlockWiseHistograms_d)
{
	int tx=threadIdx.x,bx=blockIdx.x,inx=bx*blockDim.x+tx;
	int rank;

	if(tx==0)
	{
		for(int j=inx+BLOCKSIZE-1;j>=inx;j--)
		{
			rank=BlockWiseHistograms_d[digit(array_d[j],exp)]-1;

			BlockWiseSortedArray_d[rank+inx]=array_d[j];

			BlockWiseHistograms_d[digit(array_d[j],exp)]--;
		}
	}
	__syncthreads();

	array_d[inx]=BlockWiseSortedArray_d[inx];
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

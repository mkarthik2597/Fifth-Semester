#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<time.h>

#define NUM_BINS 128

__global__ void Histogram(const int*,int*,int);
bool CheckAnswer(int*,int*,int);
void printArray(int*,int);


int main()
{	
	/* Variables to measure the overhead time*/
	clock_t start_overhead,end_overhead;
	start_overhead=clock();
	
	int nROW=12000;	
	
	int* Array=new int[nROW];
	int* HostBins=new int[NUM_BINS];
	
	srand(time(NULL));
	
	/* Populate the input array*/
	for(int i=0;i<nROW;i++)
	Array[i]=rand()%NUM_BINS;
  
	int ArraySize=nROW*sizeof(int);
	int BinSize=NUM_BINS*sizeof(int);
	
	int* Array_d, *DeviceBins;
	
	/* Allocate memory on the device*/
	
	cudaMalloc((void**)&Array_d,ArraySize);
	cudaMalloc((void**)&DeviceBins,BinSize);
	
	/* Copy data from host to device*/
	cudaMemcpy(Array_d,Array,ArraySize,cudaMemcpyHostToDevice);
	cudaMemcpy(DeviceBins,HostBins,BinSize,cudaMemcpyHostToDevice);

	/* Variables to measure device computation time*/
	clock_t start,end,total;
	start=clock();
	
	/* Kernel launch*/
	Histogram<<<ceil(nROW/128.0),128>>>(Array_d,DeviceBins,nROW);
	
	end=clock();
	total=(double)(end - start) / CLOCKS_PER_SEC;
	printf("Time taken on device: %lf\n",total);
	
	/* Copy the convoluted image to host*/
	cudaMemcpy(HostBins,DeviceBins,BinSize,cudaMemcpyDeviceToHost);
	
	end_overhead=clock();
	
	/* Verify solution*/
	if(CheckAnswer(Array,HostBins,nROW))
	printf("Solution is right!\n");
    else
	printf("Solution is wrong!\n");
	
	printf("Time spent on overhead calculations: %lf\n",(double)(end_overhead - start_overhead) / CLOCKS_PER_SEC);
	
/* 	printArray(Array,nROW);
	printArray(HostBins,NUM_BINS); */
	
	cudaFree(Array_d);
	cudaFree(DeviceBins);
}

__global__ void Histogram(const int * __restrict__ deviceInput,int *deviceBins,int inputLength)
{
	int row=blockIdx.x*blockDim.x+threadIdx.x;
	
	__shared__ int private_histo[128];
	private_histo[threadIdx.x]=0;
	__syncthreads();
	
	if(row<inputLength)
	{
	atomicAdd(&private_histo[deviceInput[row]],1);
	}
	__syncthreads();
	
	atomicAdd(&deviceBins[threadIdx.x],private_histo[threadIdx.x]);
}

/* A function to verify correctness of solution*/
bool CheckAnswer(int* Array,int* HostBins, int nROW )
{	
	int * temp =new int[NUM_BINS];
	
	clock_t start,end;
	start=clock();
	
	for(int i=0;i<NUM_BINS;i++)
	temp[i]=0;

	for(int i=0;i<nROW;i++)
	{
		temp[Array[i]]++;
	}
	
	end=clock();
	printf("Time taken on host: %lf\n",(double)(end - start) / CLOCKS_PER_SEC);
	
	for(int i=0;i<NUM_BINS;i++)
	{
		if(temp[i]!=HostBins[i])
		{
			printf("Location->%d,Expected->%d,Received->%d",i,temp[i],HostBins[i]);
			return false;
		}
	}
	return true;
}

void printArray(int* arr,int size)
{
	for(int i=0;i<size;i++)
	printf("%d\n",arr[i]);

	printf("\n");
}

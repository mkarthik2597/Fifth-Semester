#include <stdlib.h>
#include <stdio.h>

#define SIZE int(pow(2,10))
#define THREADS 1024
#define BLOCKS SIZE/THREADS

__global__ void BitonicSort(int* arr_d);
void CheckSolution(int* arr);
void PrintArray(int* arr);

int main()
{
	int* arr=new int[SIZE];
	for(int i=0;i<SIZE;i++)
	arr[i]=rand()%10;

	int* arr_d;
	cudaMalloc((void**)&arr_d,SIZE*sizeof(int));
	cudaMemcpy(arr_d,arr,SIZE*sizeof(int),cudaMemcpyHostToDevice);
	
	BitonicSort<<<BLOCKS,THREADS>>>(arr_d);

	cudaMemcpy(arr,arr_d,SIZE*sizeof(int),cudaMemcpyDeviceToHost);
	CheckSolution(arr);
}

__global__ void BitonicSort(int* arr_d)
{
	int i=threadIdx.x;
	__shared__ int Shared_arr[THREADS];
	Shared_arr[i]=arr_d[i];
	
	for(int k=2;k<=THREADS;k<<=1)
	for(int j=k>>1;j>0;j>>=1)
	{
		int ixj=i^j;
		if(i<ixj)
		{	// Sort ascending 
			if((i&k)==0)
			{
				if(Shared_arr[i]>Shared_arr[ixj])
				{
					int temp=Shared_arr[i];
					Shared_arr[i]=Shared_arr[ixj];
					Shared_arr[ixj]=temp;
				}
			}
			// Sort descending 
			else
			{
				if(Shared_arr[i]<Shared_arr[ixj])
				{
					int temp=Shared_arr[i];
					Shared_arr[i]=Shared_arr[ixj];
					Shared_arr[ixj]=temp;
				}
			}
		} 
		
		__syncthreads();
	}
	
	arr_d[i]=Shared_arr[i];
}

void CheckSolution(int* arr)
{
	int i;
	for(i=0;i<SIZE-1;i++)
	if(arr[i]>arr[i+1])
	{
		printf("Solution is Wrong!\n");
		break;
	}
	
	if(i==SIZE-1)
	printf("Solution is right!\n");
}

void PrintArray(int* arr)
{
	for(int i=0;i<SIZE;i++)
	printf("%d ",arr[i]);

	printf("\n");
}
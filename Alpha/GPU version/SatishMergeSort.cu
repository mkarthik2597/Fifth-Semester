#include<stdlib.h>
#include<stdio.h>
#include<math.h>

#define SIZE int(pow(2,10))
#define BLOCKSIZE 1024
#define NUMBLOCKS SIZE/BLOCKSIZE
#define VALUE(arr,bx,tx) arr[bx*blockDim.x+tx]

__global__ void SharedMemoryBitonicSort(int* arr_d);
__global__ void MergeSort(int*,int*);
void CheckSolution(int* arr);
void PrintArray(int* arr);
void CheckBitonicSolution(int* arr, int start, int size);

int main()
{
	int* arr=new int[SIZE];

	/* Generate distinct random numbers */
	int temp,j;
	for(int i=0;i<SIZE;i++)
	{
		temp=rand()%(2*SIZE);
		for(j=0;j<i;j++)
		{
			if(arr[j]==temp)
			{
				i--;
				break;
			}
		}

		if(j==i)
		arr[i]=temp;
	}

	int* arr_d;
	cudaMalloc((void**)&arr_d,SIZE*sizeof(int));
	cudaMemcpy(arr_d,arr,SIZE*sizeof(int),cudaMemcpyHostToDevice);

	SharedMemoryBitonicSort<<<NUMBLOCKS,BLOCKSIZE>>>(arr_d);

	int* result_d;
	cudaMalloc((void**)&result_d,SIZE*sizeof(int));

	for(int i=BLOCKSIZE;i<SIZE;i<<=1)
	{
		MergeSort<<<(SIZE/i),i>>>(arr_d,result_d);
		cudaMemcpy(arr_d,result_d,SIZE*sizeof(int),cudaMemcpyDeviceToHost);
	}

	cudaMemcpy(arr,arr_d,SIZE*sizeof(int),cudaMemcpyDeviceToHost);
	CheckSolution(arr);
}

__global__ void SharedMemoryBitonicSort(int* arr_d)
{
	int i=threadIdx.x;
	__shared__ int Shared_arr[BLOCKSIZE];
	Shared_arr[i]=arr_d[blockIdx.x*blockDim.x+i];

	for(int k=2;k<=BLOCKSIZE;k<<=1)
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

	arr_d[blockIdx.x*blockDim.x+i]=Shared_arr[i];
}

__global__ void MergeSort(int* arr_d, int* result_d)
{
	int tx=threadIdx.x,bx=blockIdx.x,key=VALUE(arr_d,bx,tx);

	/* partner is the block where the parallel binary search should take place */
	int partner=bx^1;
	int rank=0;

	/* Each thread performs a binary search in its partner block*/
	int low=0,high=blockDim.x-1;

	while(low<=high)
	{
		int mid=(low+high)/2;

		if(VALUE(arr_d,partner,mid)<key)
		{
			if(mid+1<blockDim.x && VALUE(arr_d,partner,mid+1)<key)
			low=mid+1;

			else
			{
				rank=mid+1;
				break;
			}
		}

		else
		{
			if(mid-1>=0 && VALUE(arr_d,partner,mid-1)>key)
			high=mid-1;

			else
			{
				rank=mid;
				break;
			}
		}
	}

	rank+=tx;

	if(bx<partner)
	VALUE(result_d,bx,rank)=key;
	else
	VALUE(result_d,partner,rank)=key;
}

void CheckSolution(int* arr)
{
	int i;
	for(i=0;i<SIZE-1;i++)
	if(arr[i]>arr[i+1])
	{
		printf("Solution is Wrong!\n%d\n%d %d\n",i,arr[i],arr[i+1]);
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

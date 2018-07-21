#include<stdio.h>
#include<stdlib.h>

#define SIZE int(pow(2,10))
#define BLOCKSIZE 1024
#define NUMBLOCKS SIZE/BLOCKSIZE

__global__ void BucketSort(int*,int*,int*);
__global__ void BitonicSort (int*);
__global__ void MergeSort(int*,int*,int);
__device__ void CompareSwapBitonicSort(int*, int i,int j);
__device__ void CompareSwapMergeSort(int*,int,int*,int);
__device__ void Copy(int*,int,int*,int);
__device__ void VectorMerge (int*,int*);
__device__ void InternalSort (int*);

void CheckSolution(int* arr);
void PrintArray(int*);

int main()
{
	int* arr=new int[SIZE];

	/* Generate distinct random numbers */
	int temp,j;
	for(int i=0;i<SIZE;i++)
	{
		temp=rand()% SIZE;
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

	int* BucketSortedList_d;
	cudaMalloc((void**)&BucketSortedList_d,SIZE*sizeof(int));

	int* BucketHistogram_d;
	cudaMalloc((void**)&BucketHistogram_d,NUMBLOCKS*sizeof(int));
	BucketSort<<<NUMBLOCKS,BLOCKSIZE>>>(arr_d,BucketSortedList_d,BucketHistogram_d);


	cudaMemcpy(arr,BucketSortedList_d,SIZE*sizeof(int),cudaMemcpyDeviceToHost);
	cudaMemcpy(arr_d,arr,SIZE*sizeof(int),cudaMemcpyHostToDevice);

	cudaFree(BucketSortedList_d);
	cudaFree(BucketHistogram_d);


	BitonicSort<<<NUMBLOCKS,BLOCKSIZE/4/*,BLOCKSIZE*/>>>(arr_d);
	cudaMemcpy(arr,arr_d,SIZE*sizeof(int),cudaMemcpyDeviceToHost);


	int* result_d;
	cudaMalloc((void**)&result_d,SIZE*sizeof(int));

	for(int i=4;i<BLOCKSIZE;i<<=1)
	{
		MergeSort<<<NUMBLOCKS,BLOCKSIZE/i/2>>>(arr_d,result_d,i);

		cudaMemcpy(arr,result_d,SIZE*sizeof(int),cudaMemcpyDeviceToHost);
		cudaMemcpy(arr_d,arr,SIZE*sizeof(int),cudaMemcpyHostToDevice);
	}

	CheckSolution(arr);
}

__global__ void BucketSort(int* arr_d, int* BucketSortedList_d, int* BucketHistogram_d)
{
	int tx=threadIdx.x,bx=blockIdx.x,inx=arr_d[bx*BLOCKSIZE+tx];

	if(tx==0)
	BucketHistogram_d[bx]=0;

	__syncthreads();

	int bucketIdx=inx/BLOCKSIZE;

	int bucketOffset=atomicAdd(&BucketHistogram_d[bucketIdx],1);

	BucketSortedList_d[bucketIdx*BLOCKSIZE+bucketOffset]=inx;
}

__global__ void BitonicSort (int* arr_d)
{
	int tx=threadIdx.x,bx=blockIdx.x,index=bx*BLOCKSIZE+4*tx;

	CompareSwapBitonicSort(arr_d,index,index+1);
	CompareSwapBitonicSort(arr_d,index+2,index+3);
	CompareSwapBitonicSort(arr_d,index,index+2);
	CompareSwapBitonicSort(arr_d,index+1,index+3);
	CompareSwapBitonicSort(arr_d,index+1,index+2);
}

__global__ void MergeSort(int* arr_d, int* result_d, int listSize)
{
	int index=blockIdx.x*BLOCKSIZE+2*listSize*threadIdx.x;
	int vec[2][4];

	int a=0,b=1;

	int a_idx=index,b_idx=index+listSize;

	Copy(vec[a],0,arr_d,a_idx);
	Copy(vec[b],0,arr_d,b_idx);
	a_idx+=4;
	b_idx+=4;

	int i=0;
	while(1)
	{
		VectorMerge(vec[a],vec[b]);
		InternalSort(vec[a]);
		InternalSort(vec[b]);

		Copy(result_d,index+i,vec[a],0);
		i+=4;

		Copy(vec[a],0,vec[b],0);

		if(a_idx!=index+listSize && b_idx!=index+2*listSize)
		{
			if(arr_d[a_idx]<arr_d[b_idx])
			{
				Copy(vec[b],0,arr_d,a_idx);
				a_idx+=4;
			}
			else
			{
				Copy(vec[b],0,arr_d,b_idx);
				b_idx+=4;
			}
		}
		else if(a_idx==index+listSize && b_idx!=index+2*listSize)
		{
			Copy(vec[b],0,arr_d,b_idx);
			b_idx+=4;
		}
		else if(b_idx==index+2*listSize && a_idx!=index+listSize)
		{
			Copy(vec[b],0,arr_d,a_idx);
			a_idx+=4;
		}
		else
		{
			break;
		}
	}

	Copy(result_d,index+i,vec[a],0);
}

__device__ void InternalSort (int* array)
{
	CompareSwapMergeSort(array,0,array,1);
	CompareSwapMergeSort(array,2,array,3);
	CompareSwapMergeSort(array,0,array,2);
	CompareSwapMergeSort(array,1,array,3);
	CompareSwapMergeSort(array,1,array,2);

}
__device__ void CompareSwapBitonicSort(int* arr_d, int i,int j)
{
	int temp;
	if(arr_d[i]>arr_d[j])
	{
		temp=arr_d[i];
		arr_d[i]=arr_d[j];
		arr_d[j]=temp;
	}
}

__device__ void CompareSwapMergeSort(int* vecA, int i, int* vecB, int j)
{
	int temp;
	if(vecB[j]<vecA[i])
	{
		temp=vecA[i];
		vecA[i]=vecB[j];
		vecB[j]=temp;
	}
}

__device__ void Copy(int* vec, int idx1, int* arr_d, int idx2)
{
	for(int i=0;i<4;i++)
	vec[idx1+i]=arr_d[idx2+i];
}

__device__ void VectorMerge (int* vecA, int* vecB)
{
	for(int i=0;i<4;i++)
	CompareSwapMergeSort(vecA,i,vecB,4-i-1);
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
	//if(arr[i]!=i)
	printf("%d ",arr[i]);

	printf("\n");
}

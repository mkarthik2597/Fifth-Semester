#include<stdio.h>
#include<cuda.h>
#include<time.h>
#include<sys/time.h>

#define SIZE 4096
#define bucketLength 1024

__global__ void bucketSortKernel(int *inData, long size, int *outData,int *spliter,int partition)
{
	__shared__ int localBucket[bucketLength];
	__shared__ int localCount; /* Counter to track index with a bucket */
       __shared__ int split[110];
	int tid = threadIdx.x; int blockId = blockIdx.x;
	int offset = blockDim.x;
	int  index, phase,i;
	float temp;

	if(tid == 0){
		localCount = 0;
for(i=0;i<109;i++)
               split[i]=spliter[i] ;
}

	__syncthreads();

       /* for(i=0;i<109;i++)
               split[i]=spliter[i] ;
*/	/* Block traverses through the array and buckets the element accordingly */
	if(tid < size) {

		if( blockId != partition-1) {
                     if(inData[tid]>=split[blockId] && inData[tid]<=split[blockId+1])
			index = atomicAdd(&localCount, 1);
			localBucket[index] = inData[tid];
		}
                else
                  { index=atomicAdd(&localCount,1);
                     localBucket[index]=inData[tid];

                     }

	}

	__syncthreads();




	tid = threadIdx.x;
	//Sorting the bucket using Parallel Bubble Sort
	for(phase = 0; phase < bucketLength; phase ++) {
		if(phase % 2 == 0) {
			while((tid < bucketLength) && (tid % 2 == 0)) {
				if(localBucket[tid] > localBucket[tid +1]) {
					temp = localBucket[tid];
					localBucket[tid] = localBucket[tid + 1];
					localBucket[tid + 1] = temp;
				}
				tid += offset;
			}
		}
		else {
			while((tid < bucketLength - 1) && (tid %2 != 0)) {
				if(localBucket[tid] > localBucket[tid + 1]) {
					temp = localBucket[tid];
					localBucket[tid] = localBucket[tid + 1];
					localBucket[tid + 1] = temp;
				}
				tid += offset;
			}
		}
	}
	tid = threadIdx.x;
	while(tid < bucketLength) {
		outData[(blockIdx.x * bucketLength) + tid] = localBucket[tid];
		tid += offset;
	}
}


void quicksort(int array[], int firstIndex, int lastIndex)
{
    //declaring index variables
    int pivotIndex, temp, index1, index2;

    if(firstIndex < lastIndex)
    {
        //assigning first element index as pivot element
        pivotIndex = firstIndex;
        index1 = firstIndex;
        index2 = lastIndex;

        //Sorting in Ascending order with quick sort
        while(index1 < index2)
        {
            while(array[index1] <= array[pivotIndex] && index1 < lastIndex)
            {
                index1++;
            }
            while(array[index2]>array[pivotIndex])
            {
                index2--;
            }

            if(index1<index2)
            {
                //Swapping opertation
                temp = array[index1];
                array[index1] = array[index2];
                array[index2] = temp;
            }
        }

        //At the end of first iteration, swap pivot element with index2 element
        temp = array[pivotIndex];
        array[pivotIndex] = array[index2];
        array[index2] = temp;

        //Recursive call for quick sort, with partiontioning
        quicksort(array, firstIndex, index2-1);
        quicksort(array, index2+1, lastIndex);
    }
}



int main()
{
	int *input, *output,*splitter;
	int *d_input, *d_output,*d_splitter;
	int i,p;
	float elapsedTime;
	cudaEvent_t start, stop;

	cudaEventCreate(&start, 0);
	cudaEventCreate(&stop, 0);

	/* Each block sorts one bucket */
	p=(SIZE/1024);

        splitter=(int*)malloc(sizeof(int)*p+5) ;
	input = (int *)malloc(sizeof(int) * SIZE);
	output = (int *)malloc(sizeof(int) * bucketLength * p);
	cudaMalloc((void**)&d_input, sizeof(int) * SIZE);
	cudaMalloc((void **)&d_output, sizeof(int) * bucketLength * p);
         cudaMalloc((void**)&d_splitter,sizeof(int)*p+10);
	cudaMemset(d_output, 0, sizeof(int) * bucketLength * p);

	srand(time(NULL));
	for(i = 0; i < SIZE; i++){
	input[i]=rand()%102400;
}

  splitter[0]=0;
  for(i=1;i<p;i++)
     splitter[i]=input[rand()%SIZE];

splitter[p]=SIZE;
 quicksort(splitter,0,p);

	// Printing the input array
/*	for(i = 0; i <= p; i++)
		printf("%d ", splitter[i]);
*///	printf("***********************\n");






        int numthreads=1024;
	cudaMemcpy(d_input, input, sizeof(int) * SIZE, cudaMemcpyHostToDevice);
        cudaMemcpy(d_splitter, splitter, sizeof(int) * p, cudaMemcpyHostToDevice);
 cudaEventRecord(start, 0);

	bucketSortKernel<<<p,numthreads >>>(d_input, SIZE, d_output,d_splitter,p);
	cudaEventRecord(stop, 0);
cudaMemcpy(output, d_output, sizeof(int) * bucketLength * p, cudaMemcpyDeviceToHost);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
//   cudaMemcpy(output, d_output, sizeof(int) * bucketLength * p, cudaMemcpyDeviceToHost);

/*
	//Printing the sorted array
	for(i = 0; i < range; i++) {
		for(j = 0; j < bucketLength; j++)
			if(output[i*bucketLength + j] != 0)
				printf("%0.4f ", output[i*bucketLength + j]);
	}

	printf("\n");
*/
	printf("Time :  %f ms \n", elapsedTime);

	cudaFree(d_input);
	cudaFree(d_output);
	free(input);
	free(output);

	return 0;
}

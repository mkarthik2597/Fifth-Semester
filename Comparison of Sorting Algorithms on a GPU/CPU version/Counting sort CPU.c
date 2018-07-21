#include<stdio.h>
#include<stdlib.h>

void printArray(int* arr, int size)
{
	for(int i=0;i<size;i++)
	printf("%d ",arr[i]);

	printf("\n");
}

int main()
{
	printf("Enter the number of elements in the input array:");
	int n;
	scanf("%d",&n);
	printf("Enter the range of the elements:");
	int k;
	scanf("%d",&k);
	
	int* input=malloc(n*sizeof(int));
	int* output=malloc(n*sizeof(int));
	
	/* 0 to k inclusive */
	int countArray[k+1];
	
	for(int i=0;i<=k;i++)
	countArray[i]=0;
	
	/* Histogram computation */8
	for(int i=0;i<n;i++)
	{
		input[i]=rand()%(k+1);
		countArray[input[i]]++;
	}	
	
	/* Scan the histogram */
	for(int i=1;i<=k;i++)
	countArray[i]+=countArray[i-1];

	for(int i=n-1;i>=0;i--)
	{
		output[countArray[input[i]]-1]=input[i];
		countArray[input[i]]--;
	}
	
	printArray(input,n);
	printArray(output,n);

}
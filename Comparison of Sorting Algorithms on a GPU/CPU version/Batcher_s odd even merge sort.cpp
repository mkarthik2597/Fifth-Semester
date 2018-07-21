#include<cstdio>
#include<cstdlib>
#include<cmath>

#define SIZE int(pow(2,10))

void SORT(int* arr,int low,int n);
void MERGE(int* arr,int low,int high,int n,int stride);
void COMP(int* arr,int i,int j);
void CheckSolution(int* array);
void PrintArray(int* arr,int,int);

int main()
{
	int* arr=new int[SIZE];
	for(int i=0;i<SIZE;i++)
	arr[i]=rand()%SIZE;
	
	SORT(arr,0,SIZE);
	CheckSolution(arr);
}

void SORT(int* arr,int low,int n)
{
	if(n>2)
	{
		int k=n/2;
		SORT(arr,low,k);
		SORT(arr,low+k,k);
		
		MERGE(arr,low,low+n-1,n,1);
	}
	else
	COMP(arr,low,low+1);
}

void MERGE(int* arr,int low,int high,int n,int stride)
{
	if(n>2)
	{
		MERGE(arr,low,high-stride,n/2,stride*2);
		MERGE(arr,low+stride,high,n/2,stride*2);
		
		for(int i=low+stride;i<high-stride;i+=stride)
		COMP(arr,i,i+stride);
	}
	else
	COMP(arr,low,low+stride);
}

void COMP(int* arr,int i,int j)
{
	if(arr[i]>arr[j])
	{
		int temp=arr[i];
		arr[i]=arr[j];
		arr[j]=temp;
	}
}

void CheckSolution(int* array)
{
	int i;
	for(i=0;i<SIZE-1;i++)
	if(array[i]>array[i+1])
	{
		printf("Solution is Wrong!\n");
		break;
	}

	if(i==SIZE-1)
	printf("Solution is right!\n");
}
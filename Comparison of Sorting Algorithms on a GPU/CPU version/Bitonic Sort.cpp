#include<cstdio>
#include<cstdlib>
#include<cmath>

#define SIZE int(pow(2,10))
#define ASC 0
#define DESC 1

void BitonicSort(int* array,int low,int n,int dir);
void BitonicMerge(int* array,int low,int n,int dir);
void CompareSwap(int* array,int a,int b,int dir);
void CheckSolution(int* array);

int main()
{
	int* array=new int[SIZE];
	for(int i=0;i<SIZE;i++)
	array[i]=rand()%10;

	BitonicSort(array,0,SIZE,ASC);
	CheckSolution(array);
}

void BitonicSort(int* array,int low,int n,int dir)
{
	if(n>1)
	{
		int k=n/2;
		BitonicSort(array,low,k,ASC);
		BitonicSort(array,low+k,k,DESC);
		BitonicMerge(array,low,n,dir);
	}		
}

void BitonicMerge(int* array,int low,int n,int dir)
{	
	if(n>1)
	{
		int k=n/2;
		for(int i=low;i<low+k;i++)
		CompareSwap(array,i,i+k,dir);

		BitonicMerge(array,low,k,dir);
		BitonicMerge(array,low+k,k,dir);
	}
}

void CompareSwap(int* array,int a,int b,int dir)
{
	if((dir==ASC && array[a]>array[b]) || (dir==DESC && array[a]<array[b]))
	{
		int temp=array[a];
		array[a]=array[b];
		array[b]=temp;		
	}
}

void CheckSolution(int* array)
{
	int i;
	for(i=0;i<SIZE-1;i++)
	if(array[i]>array[i+1])
	printf("Solution is Wrong!\n");

	if(i==SIZE-1)
	printf("Solution is right!\n");
}
#include<stdio.h>
#include<stdlib.h>
#include<math.h>
void printMatrix(int*);
__global__ void MatrixAddKernel(int*,int*,int*,int,int,int);
bool CheckAnswer(int*,int*,int*,int,int,int);


int main()
{	
	int nROW=1024;
	int nCOL=1025;
	int m=300;
	
	/* Dynamically allocate memory to 3 matrices. Each matrix is represented as a 1D array*/
	int* mat1=new int[nROW*m];
	int* mat2=new int[m*nCOL];
	int* product=new int[nROW*nCOL];
	
	srand(time(NULL));
	
	for(int i=0;i<nROW;i++)
	for(int j=0;j<m;j++)
	mat1[i*m+j]=rand()%4;

	for(int i=0;i<m;i++)
	for(int j=0;j<nCOL;j++)
	mat2[i*nCOL+j]=rand()%6;

	int size1=m*nROW*sizeof(int);
	int size2=m*nCOL*sizeof(int);
	int size3=nROW*nCOL*sizeof(int);	
	
	int* mat1_d, *mat2_d, *product_d;
	
	cudaMalloc((void**)&mat1_d,size1);
	cudaMalloc((void**)&mat2_d,size2);
	cudaMalloc((void**)&product_d,size3);
	
	cudaMemcpy(mat1_d,mat1,size1,cudaMemcpyHostToDevice);
	cudaMemcpy(mat2_d,mat2,size2,cudaMemcpyHostToDevice);
	
	/*Each block consists of 1024 threads. Each block of threads functions conceptually like a tile*/
	dim3 dimBlock(32,32);
	/*The x-dimension is horizontal and the y-dimension is vertical*/
	/*The x-dimension and the y-dimension depend on the no. of columns and the no. of rows respectively*/
	dim3 dimGrid(ceil(nCOL/32.0),ceil(nROW/32.0));
	MatrixAddKernel<<<dimGrid,dimBlock>>>(mat1_d,mat2_d,product_d,nROW,nCOL,m);
	
	cudaMemcpy(product,product_d,size3,cudaMemcpyDeviceToHost);
	
	if(CheckAnswer(mat1,mat2,product,nROW,nCOL,m))
	printf("Solution is right!\n");
    else
	printf("Solution is wrong!\n");
	
	cudaFree(mat1_d);
	cudaFree(mat2_d);
	cudaFree(product_d);	
}

__global__ void MatrixAddKernel(int* mat1_d, int* mat2_d, int* product_d, int nROW, int nCOL,int m)
{
	int row=blockIdx.y*blockDim.y+threadIdx.y;
	int col=blockIdx.x*blockDim.x+threadIdx.x; 
	
	/*Error checking is required because matrices need not fit in exact tiles*/
	if(row<nROW && col<nCOL)  
	{
		int product=0;
		for(int k=0;k<m;k++)
		product+=mat1_d[row*m+k]*mat2_d[k*nCOL+col];			

		product_d[row*nCOL+col]=product;
	} 
	
}

bool CheckAnswer(int* mat1,int* mat2, int* product,int nROW, int nCOL,int m)
{
	for(int i=0;i<nROW;i++)
	for(int j=0;j<nCOL;j++)
	{
		int temp=0;
		for(int k=0;k<m;k++)
		{
			temp+=mat1[i*m+k]*mat2[k*nCOL+j];
		}
		if(temp!=product[i*nCOL+j])
		return false;
	}
	return true;
}
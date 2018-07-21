#include<stdio.h>
#include<stdlib.h>
#include<math.h>
void printMatrix(int*);
__global__ void MatrixProductKernel(int*,int*,int*,int,int,int);
bool CheckAnswer(int*,int*,int*,int,int,int);
void printMatrix(int* mat);

int main()
{	
	int nROW=32;
	int nCOL=32;
	int m=73;
	
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
	MatrixProductKernel<<<dimGrid,dimBlock>>>(mat1_d,mat2_d,product_d,nROW,nCOL,m);
	
	cudaMemcpy(product,product_d,size3,cudaMemcpyDeviceToHost);
	
	if(CheckAnswer(mat1,mat2,product,nROW,nCOL,m))
	printf("Solution is right!\n");
    else
	printf("Solution is wrong!\n");

	printMatrix(mat1);
	printMatrix(mat2);
	printMatrix(product);
	
	cudaFree(mat1_d);
	cudaFree(mat2_d);
	cudaFree(product_d);	
}

__global__ void MatrixProductKernel(int* mat1_d, int* mat2_d, int* product_d, int nROW, int nCOL,int m)
{
	int row=blockIdx.y*blockDim.y+threadIdx.y;
	int col=blockIdx.x*blockDim.x+threadIdx.x;
	
	__shared__ int mat1_ds[32][32];
	__shared__ int mat2_ds[32][32];
	
	if(row<nROW && col<nCOL)
	{
		int product=0;
		for(int i=0;i<(float)m/blockDim.x;i++)
		{
			mat1_ds[threadIdx.y][threadIdx.x]=mat1_d[row*m+i*blockDim.x+threadIdx.x];
			mat2_ds[threadIdx.y][threadIdx.x]=mat2_d[(i*blockDim.y+threadIdx.y)*nCOL+col];
			__syncthreads();
			
			for(int j=0;j<blockDim.x;j++)
			if(j<m)
			product+=mat1_ds[threadIdx.y][j]*mat2_ds[j][threadIdx.x];
			__syncthreads();
			
		}
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
		{
			printf("%d %d\n",i,j);
			return false;
		}
	}
	return true;
}

void printMatrix(int* mat)
{
	for(int i=0;i<3;i++)
	{
		for(int j=0;j<3;j++)
		printf("%d ",mat[i*3+j]);
	    
		printf("\n");
	}
	printf("\n\n");
}

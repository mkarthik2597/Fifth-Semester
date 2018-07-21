#include<stdio.h>
#include<stdlib.h>
#include<math.h>
void printMatrix(int*,int);
__global__ void MatrixAddKernel(int*,int*,int*,int,int);


int main()
{	
	int nrow=1000;
	int ncol=1000;
	
	/* Dynamically allocate memory to 3 matrices. Each matrix is represented as a 1D array*/
	int* mat1=new int[nrow*ncol];
	int* mat2=new int[nrow*ncol];
	int* sum=new int[nrow*ncol];
	
	srand(time(NULL));
	
	for(int i=0;i<nrow;i++)
	for(int j=0;j<ncol;j++)
	{
		mat1[i*ncol+j]=rand()%4;
		mat2[i*ncol+j]=rand()%6;
	}

	int size=nrow*ncol*sizeof(int);
	
	int* mat1_d, *mat2_d, *sum_d;
	
	cudaMalloc((void**)&mat1_d,size);
	cudaMalloc((void**)&mat2_d,size);
	cudaMalloc((void**)&sum_d,size);
	
	cudaMemcpy(mat1_d,mat1,size,cudaMemcpyHostToDevice);
	cudaMemcpy(mat2_d,mat2,size,cudaMemcpyHostToDevice);
	
	/*Each block consists of 1024 threads. Each block of threads functions conceptually like a tile*/
	dim3 dimBlock(32,32);
	/*The x-dimension is horizontal and the y-dimension is vertical*/
	/*The x-dimension and the y-dimension depend on the no. of columns and the no. of rows respectively*/
	dim3 dimGrid(ceil(ncol/32.0),ceil(nrow/32.0));
	MatrixAddKernel<<<dimGrid,dimBlock>>>(mat1_d,mat2_d,sum_d,nrow,ncol);
	
	cudaMemcpy(sum,sum_d,size,cudaMemcpyDeviceToHost);
	
	printMatrix(mat1,ncol);
	printMatrix(mat2,ncol);
	printMatrix(sum,ncol);
	
	cudaFree(mat1_d);
	cudaFree(mat2_d);
	cudaFree(sum_d);	
}

__global__ void MatrixAddKernel(int* mat1_d, int* mat2_d, int* sum_d, int nrow, int ncol)
{
	int row=blockIdx.y*blockDim.y+threadIdx.y;
	int col=blockIdx.x*blockDim.x+threadIdx.x; 
	
	/*Error checking is required because matrices need not fit in exact tiles*/
	if(row<nrow && col<ncol)  
	sum_d[row*ncol+col]=mat1_d[row*ncol+col]+mat2_d[row*ncol+col]; 
	
}

void printMatrix(int* mat,int ncol)
{
	for(int i=0;i<3;i++)
	{
		for(int j=0;j<3;j++)
		printf("%d ",mat[i*ncol+j]);
	    
		printf("\n");
	}
	printf("\n\n");
}

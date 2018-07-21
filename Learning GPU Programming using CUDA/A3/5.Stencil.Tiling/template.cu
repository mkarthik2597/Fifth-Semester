#include "wb.h"

#define BW 16

#define MAX_VAL 255
#define Max(a, b) ((a) < (b) ? (b) : (a))
#define Min(a, b) ((a) > (b) ? (b) : (a))
#define Clamp(a, start, end) Max(Min(a, end), start)
#define value(arry, i, j, k) arry[((i)*width + (j)) * depth + (k)]

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

__global__ void stencil(unsigned char *output, unsigned char *input, int width, int height,
                        int depth) 
{
  //@@ INSERT CODE HERE
#define output(i, j, k) value(output, i, j, k)
#define input(i, j, k) value(input, i, j, k)

  	/* Store the thread dimensions on registers*/
	int bx=blockIdx.x,by=blockIdx.y,bz=blockIdx.z;
	int tx=threadIdx.x,ty=threadIdx.y,tz=threadIdx.z;
	
	/* Declare a shared memory region for each block of threads*/
	__shared__ int SharedMemBlock[BW][BW][BW];
	
	/* Find out the row and column for each thread*/
	int XIdx=by*blockDim.y+ty;
	int YIdx=bx*blockDim.x+tx;
	int ZIdx=bz*blockDim.z+tz;

	SharedMemBlock[ty][tx][tz]=input(XIdx,YIdx,ZIdx);
    __syncthreads();

	int Pvalue=0;
	
	if(ZIdx+1>=(bz+1)*BW)
	Pvalue+=input(XIdx,YIdx,ZIdx+1);
    else
	Pvalue+=SharedMemBlock[ty][tx][tz+1];

	if(ZIdx-1<bz*BW)
	Pvalue+=input(XIdx,YIdx,ZIdx-1);
    else
	Pvalue+=SharedMemBlock[ty][tx][tz-1];

	if(YIdx+1>=(by+1)*BW)
	Pvalue+=input(XIdx,YIdx+1,ZIdx);
    else
	Pvalue+=SharedMemBlock[ty][tx+1][tz];

	if(YIdx-1<by*BW)
	Pvalue+=input(XIdx,YIdx-1,ZIdx);
    else
	Pvalue+=SharedMemBlock[ty][tx-1][tz];

	if(XIdx+1>=(bx+1)*BW)
	Pvalue+=input(XIdx+1,YIdx,ZIdx);
    else
	Pvalue+=SharedMemBlock[ty+1][tx][tz];

	if(XIdx-1<(bx)*BW)
	Pvalue+=input(XIdx-1,YIdx,ZIdx);
    else
	Pvalue+=SharedMemBlock[ty-1][tx][tz];

	Pvalue-=6*input(XIdx,YIdx,ZIdx);
	output(XIdx,YIdx,ZIdx)=Clamp(Pvalue, 0, MAX_VAL);
	
#undef output
#undef input
}

static void launch_stencil(unsigned char *deviceOutputData, unsigned char *deviceInputData,
                           int width, int height, int depth) 
{
  //@@ INSERT CODE HERE
  	dim3 dimBlock(BW,BW,BW);
	dim3 dimGrid(ceil((float)width/BW),ceil((float)height/BW),ceil((float)depth/BW));
	
	stencil<<<dimGrid,dimBlock>>>(deviceOutputData,deviceInputData,width,height,depth);
}

int main(int argc, char *argv[]) {
  wbArg_t arg;
  int width;
  int height;
  int depth;
  char *inputFile;
  wbImage_t input;
  wbImage_t output;
  unsigned char *hostInputData;
  unsigned char *hostOutputData;
  unsigned char *deviceInputData;
  unsigned char *deviceOutputData;

  arg = wbArg_read(argc, argv);

  inputFile = wbArg_getInputFile(arg, 0);

  input = wbImport(inputFile);

  width  = wbImage_getWidth(input);
  height = wbImage_getHeight(input);
  depth  = wbImage_getChannels(input);

  output = wbImage_new(width, height, depth);

  hostInputData  = wbImage_getData(input);
  hostOutputData = wbImage_getData(output);

  wbTime_start(GPU, "Doing GPU memory allocation");
  cudaMalloc((void **)&deviceInputData,
             width * height * depth * sizeof(unsigned char));
  cudaMalloc((void **)&deviceOutputData,
             width * height * depth * sizeof(unsigned char));
  wbTime_stop(GPU, "Doing GPU memory allocation");

  wbTime_start(Copy, "Copying data to the GPU");
  cudaMemcpy(deviceInputData, hostInputData,
             width * height * depth * sizeof(unsigned char),
             cudaMemcpyHostToDevice);
  wbTime_stop(Copy, "Copying data to the GPU");

  wbTime_start(Compute, "Doing the computation on the GPU");
  launch_stencil(deviceOutputData, deviceInputData, width, height, depth);
  wbTime_stop(Compute, "Doing the computation on the GPU");

  wbTime_start(Copy, "Copying data from the GPU");
  cudaMemcpy(hostOutputData, deviceOutputData,
             width * height * depth * sizeof(unsigned char),
             cudaMemcpyDeviceToHost);
  wbTime_stop(Copy, "Copying data from the GPU");

  wbSolution(arg, output);

  cudaFree(deviceInputData);
  cudaFree(deviceOutputData);

  wbImage_delete(output);
  wbImage_delete(input);

  return 0;
}

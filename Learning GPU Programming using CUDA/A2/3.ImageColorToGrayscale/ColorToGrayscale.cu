/* 
The "wb.h" file(library) has been included in this code
First compile dataset_generator.cpp. You may use any no. of pixels in the x and y dimensions.
The dataset_generator will output input.ppm and output.ppm
Compile this file using "./a.out input.ppm output.pbm"

*/

#include "wb.h"

//@@ define error checking macro here.
#define errCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      printErrorLog(ERROR, "Failed to run stmt ", #stmt);                         \
      printErrorLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

//@@ INSERT CODE HERE
__global__ void ImageGrayScaleKernel(float* deviceInputImageData,float* deviceOutputImageData,int imageHeight,int imageWidth)
{
	int row=blockIdx.y*blockDim.y+threadIdx.y;
	int col=blockIdx.x*blockDim.x+threadIdx.x; 
	
	if(row<imageHeight && col<imageWidth)
	{
		int idx=row*imageWidth+col;
		float r=deviceInputImageData[3*idx];
		float g=deviceInputImageData[3*idx+1];
		float b=deviceInputImageData[3*idx+2];
		
		deviceOutputImageData[idx] = (0.21f * r + 0.71f * g + 0.07f * b);
	}
}

int main(int argc, char *argv[]) {

  int imageChannels;
  int imageWidth;
  int imageHeight;
  char *inputImageFile;
  wbImage_t inputImage;
  wbImage_t outputImage;
  float *hostInputImageData;
  float *hostOutputImageData;
  float *deviceInputImageData;
  float *deviceOutputImageData;

  /* parse the input arguments */
  //@@ Insert code here
  wbArg_t args;
  args = wbArg_read(argc, argv);

  inputImageFile = wbArg_getInputFile(args, 0);

  inputImage = wbImport(inputImageFile);

  imageWidth  = wbImage_getWidth(inputImage);
  imageHeight = wbImage_getHeight(inputImage);
  // For this lab the value is always 3
  imageChannels = wbImage_getChannels(inputImage);

  // Since the image is monochromatic, it only contains one channel
  outputImage = wbImage_new(imageWidth, imageHeight, 1);

  hostInputImageData  = wbImage_getData(inputImage);
  hostOutputImageData = wbImage_getData(outputImage);

  wbTime_start(GPU, "Doing GPU Computation (memory + compute)");

  wbTime_start(GPU, "Doing GPU memory allocation");
  cudaMalloc((void **)&deviceInputImageData,
             imageWidth * imageHeight * imageChannels * sizeof(float));
  cudaMalloc((void **)&deviceOutputImageData,
             imageWidth * imageHeight * sizeof(float));
  wbTime_stop(GPU, "Doing GPU memory allocation");

  wbTime_start(Copy, "Copying data to the GPU");
  cudaMemcpy(deviceInputImageData, hostInputImageData,
             imageWidth * imageHeight * imageChannels * sizeof(float),
             cudaMemcpyHostToDevice);
  wbTime_stop(Copy, "Copying data to the GPU");

  ///////////////////////////////////////////////////////
  wbTime_start(Compute, "Doing the computation on the GPU");
  //@@ INSERT CODE HERE
  dim3 dimBlock(32,32);
  dim3 dimGrid(ceil(imageWidth/32.0),ceil(imageHeight/32.0));
  ImageGrayScaleKernel<<<dimGrid,dimBlock>>>(deviceInputImageData,deviceOutputImageData,imageHeight,imageWidth);

  wbTime_stop(Compute, "Doing the computation on the GPU");

  ///////////////////////////////////////////////////////
  wbTime_start(Copy, "Copying data from the GPU");
  cudaMemcpy(hostOutputImageData, deviceOutputImageData,
             imageWidth * imageHeight * sizeof(float),
             cudaMemcpyDeviceToHost);
  wbTime_stop(Copy, "Copying data from the GPU");

  wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");

  wbSolution(args, outputImage);

  cudaFree(deviceInputImageData);
  cudaFree(deviceOutputImageData);

  wbImage_delete(outputImage);
  wbImage_delete(inputImage);

  return 0;
}

/* 
The "wb.h" file(library) has been included in this code
First compile dataset_generator.cpp. You may use any no. of pixels in the x and y dimensions.
The dataset_generator will output input.ppm and output.ppm
Compile this file using "./a.out input.ppm output.ppm"

*/

#include "wb.h"
#define BLUR_SIZE 5

//@@ INSERT CODE HERE
__global__ void ImageBlurKernel(float* deviceInputImageData,float* deviceOutputImageData,int imageHeight,int imageWidth)
{
	int row=blockIdx.y*blockDim.y+threadIdx.y;
	int col=blockIdx.x*blockDim.x+threadIdx.x; 
	
	/*Error checking is required because matrices need not fit in exact tiles*/
	if(row<imageHeight && col<imageWidth)  
	{
	    float pixVal = 0;
        int pixels = 0;
		
		for(int blurrow=-BLUR_SIZE;blurrow<BLUR_SIZE+1;blurrow++)
		{
			for (int blurcol = -BLUR_SIZE; blurcol < BLUR_SIZE + 1;blurcol++)
			{
				int currow = row + blurrow;
                int curcol = col + blurcol;
				
				if(currow>-1 && currow<imageHeight && curcol>-1 && curcol<imageWidth)
				{
					pixVal+=deviceInputImageData[currow*imageWidth+curcol];
					pixels++;
				}
			}
		}
		deviceOutputImageData[row*imageWidth+col]=float(pixVal)/pixels;
	}
	
}

int main(int argc, char *argv[]) {

  int imageWidth;
  int imageHeight;
  char *inputImageFile;
  wbImage_t inputImage;
  wbImage_t outputImage;
  float *hostInputImageData;
  float *hostOutputImageData;
  float *deviceInputImageData;
  float *deviceOutputImageData;
	
  wbArg_t args;
  args = wbArg_read(argc, argv);
  
  /* parse the input arguments */
  //@@ Insert code here

  inputImageFile = wbArg_getInputFile(args, 0);

  inputImage = wbImport(inputImageFile);

  // The input image is in grayscale, so the number of channels
  // is 1
  imageWidth  = wbImage_getWidth(inputImage);
  imageHeight = wbImage_getHeight(inputImage);

  // Since the image is monochromatic, it only contains only one channel
  outputImage = wbImage_new(imageWidth, imageHeight, 3);

  hostInputImageData  = wbImage_getData(inputImage);
  hostOutputImageData = wbImage_getData(outputImage);

  wbTime_start(GPU, "Doing GPU Computation (memory + compute)");

  wbTime_start(GPU, "Doing GPU memory allocation");
  cudaMalloc((void **)&deviceInputImageData,
             imageWidth * imageHeight * sizeof(float));
  cudaMalloc((void **)&deviceOutputImageData,
             imageWidth * imageHeight * sizeof(float));
  wbTime_stop(GPU, "Doing GPU memory allocation");

  wbTime_start(Copy, "Copying data to the GPU");
  cudaMemcpy(deviceInputImageData, hostInputImageData,
             imageWidth * imageHeight * sizeof(float),
             cudaMemcpyHostToDevice);
  wbTime_stop(Copy, "Copying data to the GPU");

  ///////////////////////////////////////////////////////
  wbTime_start(Compute, "Doing the computation on the GPU");
  
  dim3 dimBlock(32,32);
  dim3 dimGrid(ceil(imageWidth/32.0),ceil(imageHeight/32.0));
  ImageBlurKernel<<<dimGrid,dimBlock>>>(deviceInputImageData,deviceOutputImageData,imageHeight,imageWidth);
  
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

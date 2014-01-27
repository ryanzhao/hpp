//----------------------------------------------------------------------------
// This is 2-D Image convolution
//============================================================
// Ryan (Weiran) Zhao 
// Started: Sat,Jan 25th 2014 01:16:35 PM EST
// Modified: Mon,Jan 27th 2014 11:18:13 AM EST
//           It took about 10 hours to debug my code, because of my
//           misunderstanding of the unclear algorithm presented in lecture
//           notes
// Last Modified: Mon,Jan 27th 2014 11:21:32 AM EST
//----------------------------------------------------------------------------
#include    <wb.h>

#define wbCheck(stmt) do {                                 \
        cudaError_t err = stmt;                            \
        if (err != cudaSuccess) {                          \
            wbLog(ERROR, "Failed to run stmt ", #stmt);    \
            return -1;                                     \
        }                                                  \
    } while(0)


#define Mask_width  5
#define Mask_radius Mask_width/2
//-------------------------------------------
// assume blocks are squre in this assignment
//-------------------------------------------
#define OutTileWidth 12
#define InTileWidth (OutTileWidth+Mask_radius*2)
#define BLKSZ InTileWidth
#define Channels 3

//@@ INSERT CODE HERE
__global__ void convolImage(const float* __restrict__ mask, // mask
                           float *inImage, // input image
                           float *outImage, // output image
                           const int width, // image width
                           const int height) // image height
{
    //--------------------------------
    // figure out all kinds of indices
    //--------------------------------
    int tx=threadIdx.x;
    int ty=threadIdx.y;
    int row_o = blockIdx.y*OutTileWidth+ty;
    int col_o = blockIdx.x*OutTileWidth+tx;
    //int row_i = row_o - Mask_radius;
    //int col_i = col_o - Mask_radius;
    int row_i = row_o - 2;
    int col_i = col_o - 2;

    //----------------------
    // declare shared memory
    //----------------------
    __shared__ float shdTileImg[InTileWidth][InTileWidth][Channels];

    //-----------------------------
    // load data from global memory
    //-----------------------------
    //-------------------------------------------------------------------------
    // check boundary element
    // for this step, a picture of out image and in image would be very helpful
    //-------------------------------------------------------------------------
    if((row_i>=0) && (row_i<height) && (col_i>=0) && (col_i<width)){
        for(int i=0; i<Channels; i++) {
            shdTileImg[ty][tx][i] = 
                inImage[(row_i*width+col_i)*Channels+i];
        }
    } else{ // ghost element
        for(int i=0; i<Channels; i++) {
            shdTileImg[ty][tx][i] = 0.0;
        }
    }

    //-----
    // sync
    //-----
    __syncthreads();

    //-----------------------------------------
    // do calculation and write to output image
    //-----------------------------------------
    float sum;
    for(int i=0;i<Channels;i++) {
        sum=0.0;
        if(ty < OutTileWidth && tx < OutTileWidth) {
            for(int m = 0; m < Mask_width; m++ ) {
                for(int n = 0; n< Mask_width; n++) {
                    sum += mask[m*Mask_width+n] *
                        shdTileImg[m+ty][n+tx][i];
                }
            }
            //------------------------------------------------------
            // this test has to be taken under tx, ty < OutTileWidth
            //------------------------------------------------------
            if((row_o < height) && (col_o < width)) {
                outImage[(row_o*width+col_o)*Channels+i] = sum;
            }
        }
    }
    __syncthreads();
}


int main(int argc, char* argv[]) {
    wbArg_t arg;
    int maskRows;
    int maskColumns;
    int imageChannels;
    int imageWidth;
    int imageHeight;
    char * inputImageFile;
    char * inputMaskFile;
    wbImage_t inputImage;
    wbImage_t outputImage;
    float * hostInputImageData;
    float * hostOutputImageData;
    float * hostMaskData;
    float * deviceInputImageData;
    float * deviceOutputImageData;
    float * deviceMaskData;

    arg = wbArg_read(argc, argv); /* parse the input arguments */

    inputImageFile = wbArg_getInputFile(arg, 0);
    inputMaskFile = wbArg_getInputFile(arg, 1);

    inputImage = wbImport(inputImageFile);
    hostMaskData = (float *) wbImport(inputMaskFile, &maskRows, &maskColumns);

    assert(maskRows == 5); /* mask height is fixed to 5 in this mp */
    assert(maskColumns == 5); /* mask width is fixed to 5 in this mp */

    imageWidth = wbImage_getWidth(inputImage);
    imageHeight = wbImage_getHeight(inputImage);
    imageChannels = wbImage_getChannels(inputImage);

    outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);

    hostInputImageData = wbImage_getData(inputImage);
    hostOutputImageData = wbImage_getData(outputImage);

    wbTime_start(GPU, "Doing GPU Computation (memory + compute)");

    wbTime_start(GPU, "Doing GPU memory allocation");
    cudaMalloc((void **) &deviceInputImageData, imageWidth * imageHeight * imageChannels * sizeof(float));
    cudaMalloc((void **) &deviceOutputImageData, imageWidth * imageHeight * imageChannels * sizeof(float));
    cudaMalloc((void **) &deviceMaskData, maskRows * maskColumns * sizeof(float));
    wbTime_stop(GPU, "Doing GPU memory allocation");


    wbTime_start(Copy, "Copying data to the GPU");
    cudaMemcpy(deviceInputImageData,
               hostInputImageData,
               imageWidth * imageHeight * imageChannels * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(deviceMaskData,
               hostMaskData,
               maskRows * maskColumns * sizeof(float),
               cudaMemcpyHostToDevice);
    wbTime_stop(Copy, "Copying data to the GPU");


    wbTime_start(Compute, "Doing the computation on the GPU");
    //@@ INSERT CODE HERE
    //-------------------------------------------
    // declare grid dimension and block dimension
    //-------------------------------------------
    dim3 gridDim((imageWidth-1)/OutTileWidth+1, (imageHeight-1)/OutTileWidth+1, 1);
    dim3 blockDim(BLKSZ, BLKSZ,1);
    //------------------
    // launch the kernel
    //------------------
    convolImage<<<gridDim,blockDim>>>(deviceMaskData, deviceInputImageData, 
            deviceOutputImageData, imageWidth, imageHeight);

    wbTime_stop(Compute, "Doing the computation on the GPU");


    wbTime_start(Copy, "Copying data from the GPU");
    cudaMemcpy(hostOutputImageData,
               deviceOutputImageData,
               imageWidth * imageHeight * imageChannels * sizeof(float),
               cudaMemcpyDeviceToHost);
    wbTime_stop(Copy, "Copying data from the GPU");

    wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");

    wbSolution(arg, outputImage);

    cudaFree(deviceInputImageData);
    cudaFree(deviceOutputImageData);
    cudaFree(deviceMaskData);

    free(hostMaskData);
    wbImage_delete(outputImage);
    wbImage_delete(inputImage);

    return 0;
}

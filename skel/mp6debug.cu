//----------------------------------------------------------------------------
// This is 2-D Image convolution
//============================================================
// Ryan (Weiran) Zhao 
// Started: Sat,Jan 25th 2014 01:16:35 PM EST
// Last Modified: Sun,Jan 26th 2014 02:28:18 PM EST
//----------------------------------------------------------------------------
#include    <wb.h>

#define wbCheck(stmt) do {                                 \
        cudaError_t err = stmt;                            \
        if (err != cudaSuccess) {                          \
            printf("%s\n",cudaGetErrorString(err));        \
            return -1;                                     \
        }                                                  \
    } while(0)


#define Mask_width  5
#define Mask_radius Mask_width/2
//-------------------------------------------
// assume blocks are squre in this assignment
//-------------------------------------------
#define OutTileWidth 4
#define InTileWidth OutTileWidth+Mask_radius*2
#define BLKSZ InTileWidth
#define Channels 1
#define InTileIdx2ImgIdx(BIdx, TIdx) BIdx*OutTileWidth+TIdx-Mask_radius

//@@ INSERT CODE HERE
__global__ void convolImage(const float* __restrict__ mask, // mask
                           float* inImage, // input image
                           float* outImage, // output image
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
    int row_i = row_o - Mask_radius;
    int col_i = col_o - Mask_radius;

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
    float sum, tmp1, tmp2;
    for(int i=0;i<Channels;i++) {
        sum=0.0;
        if(ty < OutTileWidth && tx < OutTileWidth) {
            for(int m = 0; m < Mask_width; m++ ) {
                for(int n = 0; n< Mask_width; n++) {
                    tmp1 = mask[m*Mask_width+n];
                    tmp2 = shdTileImg[m+ty][n+tx][i];
                    //sum += mask[m*Mask_width+n] * shdTileImg[m+ty][n+tx][i];
                    sum += tmp1*tmp2;
                }
            }
            if(row_o < height && col_o < width) {
                outImage[(row_o*width+col_o)*Channels+i] = sum;
            }
        }
    }
}

void printMat(float* mat, int numRow, int numCol) {
    printf("============================================================\n");
    for(int i=0; i<numRow; i++) {
        for(int j=0; j<numCol; j++) {
            printf("%.2f\t", mat[i*numCol+j]);
        }
        printf("\n");
    }
}

int main(int argc, char* argv[]) {
    int maskRows = 5;
    int maskColumns = 5;
    int imageChannels = 1;
    int imageWidth = 13;
    int imageHeight = 13;
    float * hostInputImageData;
    float * hostOutputImageData;
    float * hostMaskData;
    float * deviceInputImageData;
    float * deviceOutputImageData;
    float * deviceMaskData;

    hostMaskData = (float*) malloc(sizeof(float)*maskRows*maskColumns);
    for(int i=0;i<maskRows;i++) {
        for(int j=0; j<maskColumns;j++) {
            hostMaskData[i*maskColumns+j] = 0.0/25;
        }
    }
    hostMaskData[2*maskColumns+2]=1;

    hostInputImageData = (float*) malloc(sizeof(float)*imageWidth*imageHeight);
    for(int i=0;i<imageHeight;i++) {
        for(int j=0; j<imageWidth;j++) {
            hostInputImageData[i*imageWidth+j] = i+j;
        }
    }
    hostOutputImageData = (float*) malloc(sizeof(float)*imageWidth*imageHeight);
    if(hostOutputImageData==NULL) {
        printf("host output image is null\n");
    }

    cudaMalloc((void **) &deviceInputImageData, imageWidth * imageHeight * imageChannels * sizeof(float));
    cudaMalloc((void **) &deviceOutputImageData, imageWidth * imageHeight * imageChannels * sizeof(float));
    cudaMalloc((void **) &deviceMaskData, maskRows * maskColumns * sizeof(float));


    wbCheck(cudaMemcpy(deviceInputImageData,
               hostInputImageData,
               imageWidth * imageHeight * imageChannels * sizeof(float),
               cudaMemcpyHostToDevice));
    wbCheck(cudaMemcpy(deviceMaskData,
               hostMaskData,
               maskRows * maskColumns * sizeof(float),
               cudaMemcpyHostToDevice));

    //-------------------------------------------
    // declare grid dimension and block dimension
    //-------------------------------------------
    dim3 gridDim((imageWidth-1)/OutTileWidth+1, (imageHeight-1)/OutTileWidth+1, 1);
    dim3 blockDim(BLKSZ, BLKSZ,1);
    printf("grid dim (%d, %d, %d)\n",gridDim.x, gridDim.y, gridDim.z);
    printf("block dim (%d, %d, %d)\n",blockDim.x, blockDim.y, blockDim.z);
    //------------------
    // launch the kernel
    //------------------
    convolImage<<<gridDim,blockDim>>>(deviceMaskData, deviceInputImageData, 
            deviceOutputImageData, imageWidth, imageHeight);


    wbCheck(cudaMemcpy(hostOutputImageData,deviceOutputImageData,imageWidth*imageHeight*sizeof(float), cudaMemcpyDeviceToHost));

    printMat(hostInputImageData,imageHeight, imageWidth);
    printMat(hostOutputImageData,imageHeight, imageWidth);

    cudaFree(deviceInputImageData);
    cudaFree(deviceOutputImageData);
    cudaFree(deviceMaskData);

    free(hostMaskData);
    free(hostInputImageData);
    free(hostOutputImageData);

    return 0;
}

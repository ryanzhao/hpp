//----------------------------------------------------------------------------
// Do matrix multiple C = A * B using the most basic algorithm on gpu
// The method used lauches a 2D grid
//============================================================
// Ryan (Weiran) Zhao 
// Started: Tue,Jan 14th 2014 09:21:51 PM EST
// Last Modified: Sat,Jan 18th 2014 09:50:51 PM EST
//----------------------------------------------------------------------------
#include<wb.h>

#define wbCheck(stmt) do {                                 \
        cudaError_t err = stmt;                            \
        if (err != cudaSuccess) {                          \
            wbLog(ERROR, "Failed to run stmt ", #stmt);    \
            return -1;                                     \
        }                                                  \
    } while(0)

// Compute C = A * B
__global__ void matrixMultiply(float * A, float * B, float * C,
			             int numARows, int numAColumns,
			             int numBRows, int numBColumns,
			             int numCRows, int numCColumns) {
    //@@ Insert code to implement matrix multiplication here
    //----------------------------------------------
    // convert (block, thread) index to matrix index
    //----------------------------------------------
    int xidx=blockIdx.x*blockDim.x+threadIdx.x;
    int yidx=blockIdx.y*blockDim.y+threadIdx.y;

    //-----------------------------
    // checking boundary conditions
    //-----------------------------
    if( (xidx<numCColumns) && (yidx<numCRows)) {
        float sum=0.0;
        int i;
        for(i=0;i<numAColumns;i++) {
            //---------------------------------------------------------
            // figuring out the index are tricky, better draw a picture
            //---------------------------------------------------------
            sum+=A[yidx*numAColumns+i]*B[i*numBColumns+xidx];
        }
        C[yidx*numCColumns+xidx] = sum;
    }
}

int main(int argc, char ** argv) {
    wbArg_t args;
    float * hostA; // The A matrix
    float * hostB; // The B matrix
    float * hostC; // The output C matrix
    float * deviceA;
    float * deviceB;
    float * deviceC;
    int numARows; // number of rows in the matrix A
    int numAColumns; // number of columns in the matrix A
    int numBRows; // number of rows in the matrix B
    int numBColumns; // number of columns in the matrix B
    int numCRows; // number of rows in the matrix C (you have to set this)
    int numCColumns; // number of columns in the matrix C (you have to set this)
    //---------------------
    // define some constant
    //---------------------
    int matASize; // size of matrix A in bytes
    int matBSize; // size of matrix B in bytes
    int matCSize; // size of matrix C in bytes
    //------------------------------
    // currently use 16-by-16 blocks
    //------------------------------
    int blkXDim = 32; 
    int blkYDim = 16;

    args = wbArg_read(argc, argv);

    wbTime_start(Generic, "Importing data and creating memory on host");
    hostA = (float *) wbImport(wbArg_getInputFile(args, 0), &numARows, &numAColumns);
    hostB = (float *) wbImport(wbArg_getInputFile(args, 1), &numBRows, &numBColumns);

    //@@ Set numCRows and numCColumns
    //---------------------------------------------------------------------
    // numCRows should be equal to numARows, numCColumns should be equal to
    // numBColumns
    //---------------------------------------------------------------------
    numCRows = numARows;
    numCColumns = numBColumns;
    //----------------------
    // specify matA,B,C size
    //----------------------
    matASize = sizeof(float)*numARows*numAColumns;
    matBSize = sizeof(float)*numBRows*numBColumns;
    matCSize = sizeof(float)*numCRows*numCColumns;

    //@@ Allocate the hostC matrix
    //-----------------------------
    // allocating host memory for C
    //-----------------------------
    hostC = (float *) malloc(matCSize);

    wbTime_stop(Generic, "Importing data and creating memory on host");

    wbLog(TRACE, "The dimensions of A are ", numARows, " x ", numAColumns);
    wbLog(TRACE, "The dimensions of B are ", numBRows, " x ", numBColumns);

    wbTime_start(GPU, "Allocating GPU memory.");
    //@@ Allocate GPU memory here
    //---------------------
    // Allocate deviceA,B,C
    //---------------------
    wbCheck(cudaMalloc((void**) &deviceA, matASize));
    wbCheck(cudaMalloc((void**) &deviceB, matBSize));
    wbCheck(cudaMalloc((void**) &deviceC, matCSize));

    wbTime_stop(GPU, "Allocating GPU memory.");

    wbTime_start(GPU, "Copying input memory to the GPU.");
    //@@ Copy memory to the GPU here
    //--------------------------
    // copy hostA,B to deviceA,B
    //--------------------------
    wbCheck(cudaMemcpy(deviceA, hostA, matASize, cudaMemcpyHostToDevice));
    wbCheck(cudaMemcpy(deviceB, hostB, matBSize, cudaMemcpyHostToDevice));

    wbTime_stop(GPU, "Copying input memory to the GPU.");
    
    //@@ Initialize the grid and block dimensions here
    // gridDim should be 1 grid, blockDim should be 2D
    dim3 gridDim((numCColumns-1)/blkXDim+1,(numCRows-1)/blkYDim+1,1);
    dim3 blockDim(blkXDim, blkYDim,1);
    
    wbTime_start(Compute, "Performing CUDA computation");
    //@@ Launch the GPU Kernel here
    matrixMultiply<<<gridDim,blockDim>>>(deviceA, deviceB, deviceC,
            numARows, numAColumns, numBRows, numBColumns, numCRows, numCColumns);

    cudaThreadSynchronize();
    wbTime_stop(Compute, "Performing CUDA computation");
    
    wbTime_start(Copy, "Copying output memory to the CPU");
    //@@ Copy the GPU memory back to the CPU here
    //--------------------------
    // copy deviceC to hostC
    //--------------------------
    wbCheck(cudaMemcpy(hostC, deviceC, matCSize, cudaMemcpyDeviceToHost));

    wbTime_stop(Copy, "Copying output memory to the CPU");

    wbTime_start(GPU, "Freeing GPU Memory");
    //@@ Free the GPU memory here
    //----------------
    // gpu memory free
    //----------------
    wbCheck(cudaFree(deviceA));
    wbCheck(cudaFree(deviceB));
    wbCheck(cudaFree(deviceC));

    wbTime_stop(GPU, "Freeing GPU Memory");

    wbSolution(args, hostC, numCRows, numCColumns);

    free(hostA);
    free(hostB);
    free(hostC);

    return 0;
}


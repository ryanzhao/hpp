//----------------------------------------------------------------------------
// Do mat-mat multiplication using tiled algorithm
//============================================================
// Ryan (Weiran) Zhao 
// Started: Sat,Jan 18th 2014 09:05:24 PM EST
// Last Modified: Sun,Jan 19th 2014 12:26:14 AM EST
//----------------------------------------------------------------------------
#include    <wb.h>

#define wbCheck(stmt) do {                                 \
        cudaError_t err = stmt;                            \
        if (err != cudaSuccess) {                          \
            wbLog(ERROR, "Failed to run stmt ", #stmt);    \
            return -1;                                     \
        }                                                  \
    } while(0)

// Compute C = A * B
__global__ void matrixMultiplyShared(float * A, float * B, float * C,
			             int numARows, int numAColumns,
			             int numBRows, int numBColumns,
			             int numCRows, int numCColumns) {
    //@@ Insert code to implement matrix multiplication here
    //@@ You have to use shared memory for this MP
    //-------------------------------------------------------------
    // calculate matrix element index based on threadid and blockid
    //-------------------------------------------------------------
    int mat_xidx = blockIdx.x*blockDim.x+threadIdx.x;
    int mat_yidx = blockIdx.y*blockDim.y+threadIdx.y;

    //---------------------------------------------
    // declare tile dim and  shared memory for tile
    //---------------------------------------------
    // for this problem ,tile has to be square
    const int tileDim = 8;
    __shared__ float matATile[tileDim][tileDim];
    __shared__ float matBTile[tileDim][tileDim];

    //-----------------------
    // loop through each tile
    //-----------------------
    int tileid;
    int tile_xidx; // x index with respect to tile coordinate
    int tile_yidx; // y index with respect to tile coordinate
    int tmp;
    float sum = 0.0; // sum for element in C
    for(tileid=0;tileid<(numAColumns-1)/tileDim+1;tileid++) {
        //-----------------------------------------------------------
        // each thread need to get one element from A tile and B tile
        //-----------------------------------------------------------
        tile_xidx = mat_xidx % tileDim;
        tile_yidx = mat_yidx % tileDim;

        //------------------------
        // get element from A tile
        //------------------------
        if((mat_yidx < numARows) && ((tmp = tileid * tileDim + tile_xidx) <
                    numAColumns)) {
            matATile[tile_yidx][tile_xidx] = A[mat_yidx * numAColumns + tmp];
        } else {
            matATile[tile_yidx][tile_xidx] = 0.0;
        }

        //------------------------
        // get element from B tile
        //------------------------
        if((mat_xidx < numBColumns) && ((tmp = tileid * tileDim + tile_yidx) <
                    numBRows)) {
            matBTile[tile_yidx][tile_xidx] = B[tmp * numBColumns + mat_xidx];
        } else {
            matBTile[tile_yidx][tile_xidx] = 0.0;
        }

        //------------------------
        // synchronize all threads
        //------------------------
        __syncthreads();

        //----------------------------------------------------
        // loop to calcuate partial value of each element in C
        //----------------------------------------------------
        for(tmp = 0; tmp < tileDim; tmp++) {
            sum += matATile[tile_yidx][tmp] *
                matBTile[tmp][tile_xidx];
        }

        //------------------------
        // synchronize all threads
        //------------------------
        __syncthreads();
    }
    //-----------------
    // write value to C
    //-----------------
    //-----------------------------------------
    // made a stupid mistake here, do remember 
    //to test the boundary conditions
    //-----------------------------------------
    if( mat_yidx < numCRows && mat_xidx < numCColumns) {
        C[mat_yidx * numCColumns + mat_xidx] = sum;
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
    //-------------------------
    // variables to save typing
    //-------------------------
    int matASize; // size of matrix A in bytes
    int matBSize; // size of matrix B in bytes
    int matCSize; // size of matrix C in bytes
    int blkXDim = 8; // x dim of block
    int blkYDim = 8; // y dim of block

    args = wbArg_read(argc, argv);

    wbTime_start(Generic, "Importing data and creating memory on host");
    hostA = (float *) wbImport(wbArg_getInputFile(args, 0), &numARows, &numAColumns);
    hostB = (float *) wbImport(wbArg_getInputFile(args, 1), &numBRows, &numBColumns);
    //@@ Set numCRows and numCColumns
    //-----------------------------------------------------------------
    // numCRows should be equal to numARows, numCColumns to numBColumns
    //-----------------------------------------------------------------
    numCRows = numARows;
    numCColumns = numBColumns;
    matASize = sizeof(float) * numARows * numAColumns;
    matBSize = sizeof(float) * numBRows * numBColumns;
    matCSize = sizeof(float) * numCRows * numCColumns;

    //@@ Allocate the hostC matrix
    hostC = (float*) malloc(matCSize);
    
    wbTime_stop(Generic, "Importing data and creating memory on host");

    wbLog(TRACE, "The dimensions of A are ", numARows, " x ", numAColumns);
    wbLog(TRACE, "The dimensions of B are ", numBRows, " x ", numBColumns);

    wbTime_start(GPU, "Allocating GPU memory.");
    //@@ Allocate GPU memory here
    //-----------------------
    // allocate memory on gpu
    //-----------------------
    wbCheck(cudaMalloc((void**) &deviceA, matASize));
    wbCheck(cudaMalloc((void**) &deviceB, matBSize));
    wbCheck(cudaMalloc((void**) &deviceC, matCSize));

    wbTime_stop(GPU, "Allocating GPU memory.");

    wbTime_start(GPU, "Copying input memory to the GPU.");
    //@@ Copy memory to the GPU here
    //--------------------------------
    // memory copy from host to device
    //--------------------------------
    wbCheck(cudaMemcpy(deviceA, hostA, matASize, cudaMemcpyHostToDevice));
    wbCheck(cudaMemcpy(deviceB, hostB, matBSize, cudaMemcpyHostToDevice));

    wbTime_stop(GPU, "Copying input memory to the GPU.");
    
    //@@ Initialize the grid and block dimensions here
    //-------------------------
    // grid and block dimension
    //-------------------------
    dim3 gridDim((numCColumns-1)/blkXDim+1, (numCRows-1)/blkYDim+1, 1);
    dim3 blockDim(blkXDim, blkYDim,1);
    
    wbTime_start(Compute, "Performing CUDA computation");
    //@@ Launch the GPU Kernel here
    //-------------
    // lauch kernel
    //-------------
    matrixMultiplyShared<<<gridDim, blockDim>>>(deviceA, deviceB, deviceC,
            numARows, numAColumns, numBRows, numBColumns, numCRows, numCColumns);

    cudaThreadSynchronize();
    wbTime_stop(Compute, "Performing CUDA computation");
    
    wbTime_start(Copy, "Copying output memory to the CPU");
    //@@ Copy the GPU memory back to the CPU here
    //--------------------------------
    // memory copy from device to host
    //--------------------------------
    wbCheck(cudaMemcpy(hostC, deviceC, matCSize, cudaMemcpyDeviceToHost));

    wbTime_stop(Copy, "Copying output memory to the CPU");

    wbTime_start(GPU, "Freeing GPU Memory");
    //@@ Free the GPU memory here
    //----------------
    // free gpu memory
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


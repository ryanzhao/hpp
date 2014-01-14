// MP 1
//----------------------------------------------------------------------------
// For this vector addition code, I am using 1D blocks/grid and 1D threads/block
// Ryan (Weiran) Zhao 
//============================================================
// Started: Mon,Jan 13th 2014 08:42:06 PM EST
// Last Modified: Mon,Jan 13th 2014 09:18:23 PM EST
//----------------------------------------------------------------------------
#include<wb.h>

__global__ void vecAdd(float * in1, float * in2, float * out, int len) {
    //@@ Insert code to implement vector addition here

    //-------------------------------------------------
    // calculate data index from block and thread index
    //-------------------------------------------------
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if(i<len) {
        out[i] = in1[i] + in2[i];
    }
}

int main(int argc, char ** argv) {
    wbArg_t args;
    int inputLength;
    float * hostInput1;
    float * hostInput2;
    float * hostOutput;
    float * deviceInput1;
    float * deviceInput2;
    float * deviceOutput;

    args = wbArg_read(argc, argv);

    wbTime_start(Generic, "Importing data and creating memory on host");
    hostInput1 = (float *) wbImport(wbArg_getInputFile(args, 0), &inputLength);
    hostInput2 = (float *) wbImport(wbArg_getInputFile(args, 1), &inputLength);
    hostOutput = (float *) malloc(inputLength * sizeof(float));
    wbTime_stop(Generic, "Importing data and creating memory on host");

    wbLog(TRACE, "The input length is ", inputLength);

	wbTime_start(GPU, "Allocating GPU memory.");
    //@@ Allocate GPU memory here
    //----------------------------------
    // use error code as a good practice
    //----------------------------------
    cudaError_t err;
    //------------------------------------------------------
    // allocate deviceInput1, deviceInput2, and deviceOutput
    //------------------------------------------------------
    if( (err=cudaMalloc((void**) &deviceInput1, sizeof(float)*inputLength)) != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err),
                __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }
    if( (err=cudaMalloc((void**) &deviceInput2, sizeof(float)*inputLength)) != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err),
                __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }
    if( (err=cudaMalloc((void**) &deviceOutput, sizeof(float)*inputLength)) != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err),
                __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }

    wbTime_stop(GPU, "Allocating GPU memory.");

    wbTime_start(GPU, "Copying input memory to the GPU.");
    //@@ Copy memory to the GPU here
    //------------------------------------
    // copy hostInput1,2 to deviceInput1,2
    //------------------------------------
    if( (err=cudaMemcpy(deviceInput1, hostInput1,sizeof(float)*inputLength,
                    cudaMemcpyHostToDevice)) != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err),
                __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }
    if( (err=cudaMemcpy(deviceInput2, hostInput2,sizeof(float)*inputLength,
                    cudaMemcpyHostToDevice)) != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err),
                __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }

    wbTime_stop(GPU, "Copying input memory to the GPU.");
    
    //@@ Initialize the grid and block dimensions here
    //------------------------------------
    // use 1D dimension for grid and block
    //------------------------------------
    dim3 gridDim(1,1,1);
    dim3 blockDim(inputLength,1,1);
    
    wbTime_start(Compute, "Performing CUDA computation");
    //@@ Launch the GPU Kernel here
    vecAdd<<<gridDim,blockDim>>>(deviceInput1,deviceInput2,deviceOutput,inputLength);


    cudaThreadSynchronize();
    wbTime_stop(Compute, "Performing CUDA computation");
    
    wbTime_start(Copy, "Copying output memory to the CPU");
    //@@ Copy the GPU memory back to the CPU here
    //--------------------------
    // copy data from gpu to cpu
    //--------------------------
    if( (err=cudaMemcpy(hostOutput, deviceOutput,sizeof(float)*inputLength,
                    cudaMemcpyDeviceToHost)) != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err),
                __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }

    wbTime_stop(Copy, "Copying output memory to the CPU");

    wbTime_start(GPU, "Freeing GPU Memory");
    //@@ Free the GPU memory here
    //-------------------------------------
    // free deviceInput1,2 and deviceOutput
    //-------------------------------------
    if( (err=cudaFree(deviceInput1)) != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err),
                __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }
    if( (err=cudaFree(deviceInput2)) != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err),
                __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }
    if( (err=cudaFree(deviceOutput)) != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err),
                __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }

    wbTime_stop(GPU, "Freeing GPU Memory");

    wbSolution(args, hostOutput, inputLength);

    free(hostInput1);
    free(hostInput2);
    free(hostOutput);

    return 0;
}


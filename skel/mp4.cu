// MP 4 Reduction
// Given a list (lst) of length n
// Output its sum = lst[0] + lst[1] + ... + lst[n-1];

#include    <wb.h>

#define BLOCK_SIZE 512 //@@ You can change this

#define wbCheck(stmt) do {                                 \
        cudaError_t err = stmt;                            \
        if (err != cudaSuccess) {                          \
            wbLog(ERROR, "Failed to run stmt ", #stmt);    \
            return -1;                                     \
        }                                                  \
    } while(0)

__global__ void total(float * input, float * output, int len) {
    //---------------------------------
    // block thread index to data index
    //---------------------------------
    int tid = threadIdx.x;
    // start position of gobal "input" data for each thread block
    int start = blockIdx.x*blockDim.x*2;
    //-----------------------
    // allocate shared memory
    //-----------------------
    __shared__ float shdVect[BLOCK_SIZE*2];
    //@@ Load a segment of the input vector into shared memory
    if((start+tid) < len) {
        shdVect[tid]=input[start+tid];
        if((start+blockDim.x+tid) < len) {
            shdVect[blockDim.x+tid] = input[start+blockDim.x+tid];
        } 
    } 

    //@@ Traverse the reduction tree
    int stride = blockDim.x;
    while(stride > 0) {
        __syncthreads();
        //---------------------------------------
        // participating threads halved each time
        //---------------------------------------
        if((tid<stride) && ((start+tid+stride) < len)) {
            shdVect[tid] += shdVect[tid+stride];
        }
        stride/=2;
    }
    //@@ Write the computed sum of the block to the output vector at the correct index
    if(tid==0) {
        output[blockIdx.x] = shdVect[tid];
    }
}

int main(int argc, char ** argv) {
    int ii;
    wbArg_t args;
    float * hostInput; // The input 1D list
    float * hostOutput; // The output list
    float * deviceInput;
    float * deviceOutput;
    int numInputElements; // number of elements in the input list
    int numOutputElements; // number of elements in the output list

    args = wbArg_read(argc, argv);

    wbTime_start(Generic, "Importing data and creating memory on host");
    hostInput = (float *) wbImport(wbArg_getInputFile(args, 0), &numInputElements);

    numOutputElements = numInputElements / (BLOCK_SIZE<<1);
    if (numInputElements % (BLOCK_SIZE<<1)) {
        numOutputElements++;
    }
    hostOutput = (float*) malloc(numOutputElements * sizeof(float));

    wbTime_stop(Generic, "Importing data and creating memory on host");

    wbLog(TRACE, "The number of input elements in the input is ", numInputElements);
    wbLog(TRACE, "The number of output elements in the input is ", numOutputElements);

    wbTime_start(GPU, "Allocating GPU memory.");
    //@@ Allocate GPU memory here
    //-----------------------------------------
    // allocate input and output buffer for gpu
    //-----------------------------------------
    wbCheck(cudaMalloc((void**) &deviceInput, sizeof(float)*numInputElements));
    wbCheck(cudaMalloc((void**) &deviceOutput, sizeof(float)*numOutputElements));

    wbTime_stop(GPU, "Allocating GPU memory.");

    wbTime_start(GPU, "Copying input memory to the GPU.");
    //@@ Copy memory to the GPU here
    //--------------------------------
    // copy memory from host to device
    //--------------------------------
    wbCheck(cudaMemcpy(deviceInput, hostInput, sizeof(float)*numInputElements, cudaMemcpyHostToDevice));

    wbTime_stop(GPU, "Copying input memory to the GPU.");
    //@@ Initialize the grid and block dimensions here
    // specify grid and block dimension
    dim3 gridDim((numInputElements-1)/(2*BLOCK_SIZE)+1,1,1);
    dim3 blockDim(BLOCK_SIZE,1,1);

    wbTime_start(Compute, "Performing CUDA computation");
    //@@ Launch the GPU Kernel here
    total<<<gridDim,blockDim>>>(deviceInput, deviceOutput, numInputElements);


    cudaDeviceSynchronize();
    wbTime_stop(Compute, "Performing CUDA computation");

    wbTime_start(Copy, "Copying output memory to the CPU");
    //@@ Copy the GPU memory back to the CPU here
    //-----------------------------
    // copy memory from gpu to host
    //-----------------------------
    wbCheck(cudaMemcpy(hostOutput, deviceOutput, sizeof(float)*numOutputElements, cudaMemcpyDeviceToHost));
    for(int i=0;i<numOutputElements;i++) {
        printf("data %f\n",hostOutput[i]);
    }

    wbTime_stop(Copy, "Copying output memory to the CPU");

    /********************************************************************
     * Reduce output vector on the host
     * NOTE: One could also perform the reduction of the output vector
     * recursively and support any size input. For simplicity, we do not
     * require that for this lab.
     ********************************************************************/
    for (ii = 1; ii < numOutputElements; ii++) {
        hostOutput[0] += hostOutput[ii];
    }

    wbTime_start(GPU, "Freeing GPU Memory");
    //@@ Free the GPU memory here
    //-----------------
    // free cuda memory
    //-----------------
    wbCheck(cudaFree(deviceInput));
    wbCheck(cudaFree(deviceOutput));

    wbTime_stop(GPU, "Freeing GPU Memory");

    wbSolution(args, hostOutput, 1);

    free(hostInput);
    free(hostOutput);

    return 0;
}


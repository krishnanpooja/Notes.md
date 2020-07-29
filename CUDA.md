**CUDA- Compute Device Unified Architecture**

Setup inputs on the host (CPU-accessible memory)
- Allocate memory for outputs on the host
- Allocate memory for inputs on the GPU
- Allocate memory for outputs on the GPU
- Copy inputs from host to GPU
- Start GPU kernel
- Copy output from GPU to CPU

-- .cu/.cuh is compiled by nvcc to produce a .o file

**Example:**
A basic kernel in CUDA starts with the keyword __global__, meaning the function is available for both host and device.

```
/* Define a kernel */
__global__ 
void cudaAddVectors(float *a, float *b, float *c) {
  unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  c[index] = a[index] + b[index]
}

/* Launch a kernel */
cudaAddVectors<<<grid_dim, block_dim>>>(a_mem, b_mem, c_mem);
```

**threadIdx**

threadIdx is a unique thread identifier within a block. blockIdx is a unique block identifier with in a grid. blockDim is the number of threads defined in a block, this number is defined in the second parameter of kernel launch (in <<< >>>). 

grid
|
block-blockDim,blockIdx
|
thread - threadIdx

**Example2**
```
void cudaAddVectors(const float *a,const float *b,float *c,size)
{
 float *dev_a;
 float *dev_b;
 float *dev_c;
 
 cudaMalloc((void**)&dev_a,size*sizeof(float));
 cudaMemcpy(dev_a,a,size*sizeof(float),cudaMemcpyHostToDevice);
 
 cudaMalloc((void**)&dev_b,size*sizeof(float));
 cudaMemcpy(dev_b,b,size*sizeof(float),cudaMemcpyHostToDevice);

 
 cudaMalloc((void**)&dev_c,size*sizeof(float));
 
 const unsigned int threadsperblck=512
 const unsigned int blocks = ceil(size/float(threadsperblck))
 
 cudaAddVectors<<threadsperblck,block>>(dev_a,dev_b,dev_c);
 
 cudaMemcpy(c,dev_c,size*sizeof(float),cudaMemcpyDeviceToHost)
 
 cudaFree(dev_a);
 cudaFree(dev_b);
 cudaFree(dev_c);
 }
 ```
 
 (SIMD − single instruction, multiple-data)
 GPUs are designed for data intensive applications
 CPU-host and GPU- device
 a typical GPU comes with its own global memory (DRAM- Dynamic Random Access Memory)
 
 __global__---> executable on CPU n callable from GPU
 __device__--> exec-GPU call-CPU
 __host__--> exec-GPU call-GPU
 
 example:
 Let us consider an example to understand the concept explained above. Consider an image, which is 76 pixels along the x axis, and 62 pixels along the y axis. Our aim is to convert the image from sRGB to grayscale. We can calculate the total number of pixels by multiplying the number of pixels along the x axis with the total number along the y axis that comes out to be 4712 pixels. Since we are mapping each thread with each pixel, we need a minimum of 4712 pixels. Let us take number of threads in each direction to be a multiple of 4. So, along the x axis, we will need at least 80 threads, and along the y axis, we will need at least 64 threads to process the complete image. We will ensure that the extra threads are not assigned any work.

Thus, we are launching 5120 threads to process a 4712 pixels image. You may ask, why the extra threads? The answer to this question is that keeping the dimensions as multiple of 4 has many benefits that largely offsets any disadvantages that result from launching extra threads. This is explained in a later section).

Now, we have to divide these 5120 threads into grids and blocks. Let each block have 256 threads. If so, then one possibility that of the dimensions each block are: (16,16,1). This means, there are 16 threads in the x direction, 16 in the y direction, and 1 in the z direction. We will be needing 5 blocks in the x direction (since there are 80 threads in total along the x axis), and 4 blocks in y direction (64 threads along the y axis in total), and 1 block in z direction. So, in total, we need 20 blocks. In a nutshell, the grid dimensions are (5,4,1) and the block dimensions are (16,16,1). The programmer needs to specify these values in the program. This is shown in the figure above.

dim3 dimBlock(5,4,1) − To specify the grid dimensions

dim3 dimGrid(ceil(n/16.0),ceil(m/16.0),1) − To specify the block dimensions.

kernelName<<<dimGrid,dimBlock>>>(parameter1, parameter2, ...) − Launch the actual kernel.
grid-->block--->thread

n is the number of pixels in the x direction, and m is the number of pixels in the y direction. ‘ceil’ is the regular ceiling function. We use it because we never want to end up with less number of blocks than required. dim3 is a data structure, just like an int or a float. dimBlock and dimGrid are variables names. The third statement is the kernel launch statement. ‘kernelName’ is the name of the kernel function, to which we pass the parameters: parameter1, parameter2, and so on. <<<>>> contain the dimensions of the grid and the block.

**Example3**
```
Matrix Multiplication
#include <math.h>
#include <iostream>
#include "cuda_runtime.h"
#include "kernel.h"
#include <stdlib.h>

using namespace std;

__global__ void matrixMultiplicationKernel(float* A, float* B, float* C, int N) {

    int ROW = blockIdx.y*blockDim.y+threadIdx.y;
    int COL = blockIdx.x*blockDim.x+threadIdx.x;

    float tmpSum = 0;

    if (ROW < N && COL < N) {
        // each thread computes one element of the block sub-matrix
        for (int i = 0; i < N; i++) {
            tmpSum += A[ROW * N + i] * B[i * N + COL];
        }
    }
    C[ROW * N + COL] = tmpSum;
}


void matrixMultiplication(float *A, float *B, float *C, int N){

    // declare the number of blocks per grid and the number of threads per block
    // use 1 to 512 threads per block
    dim3 threadsPerBlock(N, N);
    dim3 blocksPerGrid(1, 1);
        if (N*N > 512){
            threadsPerBlock.x = 512;
            threadsPerBlock.y = 512;
            blocksPerGrid.x = ceil(double(N)/double(threadsPerBlock.x));
            blocksPerGrid.y = ceil(double(N)/double(threadsPerBlock.y));
        }

    matrixMultiplicationKernel<<<(N/512,N/512),(512,512)>>>(A, B, C, N);
}
```
__syncthreads() to synchronize threads. When the method is encountered in the kernel, all threads in a block will be blocked at the calling location until each of them reaches the location.
If an if-then-else statement is present inside the kernel, then either all the threads will take the if path, or all the threads will take the else path. This is implied. As all the threads of a block have to execute the sync method call, if threads took different paths, then they will be blocked forever.

- The host is the CPU available in the system. The system memory associated with the CPU is called host memory. The GPU is called a device and GPU memory likewise called device memory.
  
- To execute any CUDA program, there are three main steps:
  1. Copy the input data from host memory to device memory, also known as host-to-device transfer.
  2. Load the GPU program and execute, caching data on-chip for performance.
  3. Copy the results from device memory to host memory, also called device-to-host transfer.

#### CUDA kernel and thread hierarchy
- CUDA kernel is a function that gets executed on GPU
- Every CUDA kernel starts with a __global__ declaration specifier. Programmers provide a unique global ID to each thread by using built-in variables.
- A group of threads is called a CUDA block. CUDA blocks are grouped into a grid. A kernel is executed as a grid of blocks of threads
  threads->blocks->grids

  <img width="513" alt="image" src="https://github.com/krishnanpooja/Notes.md/assets/8016149/7a6f2de8-8dd8-41ad-b10d-ebac19858afb">

- Each CUDA block is executed by one streaming multiprocessor(SM)
- 

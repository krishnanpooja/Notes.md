## Distributed Training

#### Types of Distributed training
__Data parallelism__  - models are replicated onto different accelerators(GPU/TPU) and data is split between them
__Model parallelism__ - model too large then divide into partitions and assign diff partitions to diff accelerators

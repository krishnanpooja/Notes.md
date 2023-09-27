## Distributed Training

#### Types of Distributed training
__Data parallelism__  - models are replicated onto different accelerators(GPU/TPU) and data is split between them

__Model parallelism__ - model too large then divide into partitions and assign diff partitions to diff accelerators
model paralllelism is too complex compared to data


#### Data Parallelism
<img width="827" alt="image" src="https://github.com/krishnanpooja/Notes.md/assets/8016149/71820206-3b31-4ce0-b239-8c7f7c59f909">

#### tf.distribute.strategy
<img width="859" alt="image" src="https://github.com/krishnanpooja/Notes.md/assets/8016149/5d195cf5-e450-4dab-8e3b-b25c5d5e519d">

##### One device 

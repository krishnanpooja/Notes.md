## Distributed Training

#### Types of Distributed training
__Data parallelism__  - models are replicated onto different accelerators(GPU/TPU) and data is split between them

__Model parallelism__ - model too large then divide into partitions and assign diff partitions to diff accelerators
model paralllelism is too complex compared to data


#### Data Parallelism
<img width="827" alt="image" src="https://github.com/krishnanpooja/Notes.md/assets/8016149/71820206-3b31-4ce0-b239-8c7f7c59f909">

- There are two basic styles of distributed training using data parallelism. In synchronous training, each worker trains on its current mini batch of data, applies its own updates, communicates out its updates to the other workers. And waits to receive and apply all of the updates from the other workers before proceeding to the next mini batch. And all-reduce algorithm is an example of this. 
-  In asynchronous training, all workers are independently training over their mini batch of data and updating variables asynchronously. Asynchronous training tends to be more efficient, but can be more difficult to implement.
  
#### tf.distribute.strategy
<img width="859" alt="image" src="https://github.com/krishnanpooja/Notes.md/assets/8016149/5d195cf5-e450-4dab-8e3b-b25c5d5e519d">

##### One device 
One device strategy will place any variables created in its scope on the specified device. Input distributed through this strategy will be prefetched to the specified device. Moreover, any functions called via strategy.run will also be placed on the specified device as well. Typical usage of this strategy could be testing your code with tf.distribute.Strategy API before switching to other strategies which actually distribute to multiple devices and machines. 

#### Mirrored strategy
- Mirrored strategy supports synchronous distributed training on multiple GPUs on one machine. 
- It creates one replica per GPU device. Each variable in the model is mirrored across all the other replicas. Together these variables form a single conceptual variable called a mirrored variable. These variables are kept in sync with each other by applying identical updates.
- Efficient all-reduce algorithms are used to communicate the variable updates across the devices. All-reduce aggregates tensors across all the devices by adding them up and makes them available on each device.
- It's a fused algorithm that is very efficient and can reduce the overhead of synchronization significantly.

#### Parameter service strategy
The parameter service strategy is a common asynchronous data parallel method to scale up model training on multiple machines. A parameter server training cluster consists of workers and parameter servers. Variables are created on parameter servers, and they are read and updated by workers in each step. By default, workers read and update these variables independently without synchronizing with each other. This is why sometimes parameter server style training is also referred to as asynchronous training. 

### Fault Tolerance
- This allows you to recover from a failure incurred by preempting workers. This can be done by preserving the training state in the distributed file system. Since all the workers are kept in sync in terms of training epochs and steps, other workers would need to wait for the failed or preempted worker to restart in order to continue.
- his allows you to recover from a failure incurred by preempting workers. This can be done by preserving the training state in the distributed file system. Since all the workers are kept in sync in terms of training epochs and steps, other workers would need to wait for the failed or preempted worker to restart in order to continue. 

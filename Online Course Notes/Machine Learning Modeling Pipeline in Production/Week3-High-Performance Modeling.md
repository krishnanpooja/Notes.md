# Distributed Training

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

### Multi-worker Configuration
Now let's enter the world of multi-worker training. In TensorFlow, the TF_CONFIG environment variable is required for training on multiple machines, each of which possibly has a different role. TF_CONFIG is a JSON string used to specify the cluster configuration on each worker that is part of the cluster.

There are two components of TF_CONFIG: cluster and task.

Let's dive into how they are used:

__cluster:__

It is the same for all workers and provides information about the training cluster, which is a dict consisting of different types of jobs such as worker.

In multi-worker training with MultiWorkerMirroredStrategy, there is usually one worker that takes on a little more responsibility like saving checkpoint and writing summary file for TensorBoard in addition to what a regular worker does.

Such a worker is referred to as the chief worker, and it is customary that the worker with index 0 is appointed as the chief worker (in fact this is how tf.distribute.Strategy is implemented).

__task:__

Provides information of the current task and is different on each worker. It specifies the type and index of that worker.
Here is an example configuration:
```
tf_config = {
    'cluster': {
        'worker': ['localhost:12345', 'localhost:23456']
    },
    'task': {'type': 'worker', 'index': 0}
}
```

Since you set the task type to "worker" and the task index to 0, this machine is the first worker and will be appointed as the chief worker.

```
strategy = tf.distribute.MultiWorkerMirroredStrategy()

# Implementing distributed strategy via a context manager
with strategy.scope():
  multi_worker_model = mnist.build_and_compile_cnn_model()
```

The distribution strategy's scope dictates how and where the variables are created, and in the case of MultiWorkerMirroredStrategy, the variables created are MirroredVariables, and they are replicated on each of the workers.

# High Performance Ingestion
__Why is this required?__
<img width="928" alt="image" src="https://github.com/krishnanpooja/Notes.md/assets/8016149/d19d8cb8-2ea3-4100-b459-d7ae1794ca57">

__Tensorflow Input Pipeline: tf.data__
__parallel interleave__

<img width="517" alt="image" src="https://github.com/krishnanpooja/Notes.md/assets/8016149/cddb5780-ffa4-4c68-8760-f2ce14db1e92">

- Use AUTOTUNE to set parallelism automatically

- __Parallel mapping__

 <img width="401" alt="image" src="https://github.com/krishnanpooja/Notes.md/assets/8016149/8141739f-2a58-4eb9-bfe8-7e0ff771ec71">

#### Training large models

<img width="862" alt="image" src="https://github.com/krishnanpooja/Notes.md/assets/8016149/5e65fcdc-9380-45c4-8a75-c68a50f47b9e">

- Pipeline parallelism from google-GPipe and __Microsoft-PipeDream__ helps improve the parallel execution of models
- They integrate both model and data parallelism
- they divide mini-batch into micro-batches
- diff workers work on diff miro batches in parallel
- allows ML models to have significantly more params
- GPipe receives as input an architecture of a neural network, a mini-batch size and the number of hardware devices that will be available for the calculation. It then automatically divides the network layers into stages and the mini-batches into micro-batches, spreading them across the devices. To divide the model into key stages, GPipe estimates the cost of each layer given its activation function and the content of the training data. GPipe attempts to maximize memory allocation for model parameters.
 



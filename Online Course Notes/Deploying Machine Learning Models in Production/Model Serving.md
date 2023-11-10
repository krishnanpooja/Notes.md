- Training a good machine learning model is only the first part. You do need to make your model available to your end-users, and you do this by either providing access to the model on your server,

- Model Serving Patterns contains model, an interpreter and input data.
- Model can be trained on model decay or online learning or dynamic learning for something like time series data
- Important metrics for online inference:
   1. Latency- delay between users action and response
   2. througput- num of successfully requests served per unit time
   3. cost - cost of CPU, GPU and cache

### Why Docker?
Docker is an amazing tool that allows you to ship your software along with all of its dependencies. This is great because it enables you to run software even without installing the required interpreters or compilers for it to run.

Let's use an example to explain this better:

Suppose you trained a Deep Learning model using Python along with some libraries such as Tensorflow or JAX. For this you created a virtual environment in your local machine. Everything works fine but now you want to share this model with a colleague of yours who does not have Python installed, much less any of the required libraries.

In a pre-Docker world your colleague would have to install all of this software just to run your model. Instead by installing Docker you can share a Docker image that includes all of your software and that will be all that is needed.

### Some key concepts
You just read about Docker images and might be wondering what they are. Now you will be introduced to three key concepts to understand how Docker works. These are Dockerfiles, images and containers, and will be explained in this order as each one uses the previous ones.

__Dockerfile__: This is a special file that contains all of the instructions required to build an image. These instructions can be anything from "install Python version 3.7" to "copy my code inside the image".

__Image__: This refers to the collection of all your software in one single place. Using the previous example, the image will include Python, Tensorflow, JAX and your code. This will be achieved by setting the appropriate instructions within the Dockerfile.

__Container__: This a running instance of an image. Images by themselves don't do much aside from saving the information of your code and its dependencies. You need to run a container out of them to actually run the code within. Containers are usually meant to perform a single task but they can be used as runtimes to run software that you haven't installed.



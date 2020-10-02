## Course 1 

### Week 1 - Introduction

**What's new in TF2** https://www.tensorflow.org/api_docs/python/tf

TF1 you need to set variables, initialize, compile etc. This sets up a graph on how i/p and o/p are connected. The graph is then run in a session.

**Major developments in TF2:**
1. Eager execution by default. The variables can be used straight away. Graph compilation still exists but not needed until you want to make low-level change.
2. tf.keras as the high-level API
3. API cleanup - functions in different places doing same thing are removed.

**Install Docker**
sudo docker run hello-world


## Week 2 - The Sequential model API

Keras.io

```
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential([Dense(64,activatiion='relu',input_shape=(784,)])
```
or

```
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten,Dense

model = Sequential()
model.add(Flatten(input_shape=(28,28)))
model.add(Dense(64,activatiion='relu'))
```
**To compute the number of trainable parameters in FFNN:**

num_params = connections between layers + biases in every layer
           = (i×h_1 +h_1*h_2+....h_n+ h_n×o) + (h_1+....h_n+o)
           
**Print the model summary**
model.summary()

**Convolutional layer**

```
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten,Dense,Conv2D,MaxPooling2D

model = Sequential()
model.add(Conv2D(16,(3,3),activation='relu',input_shape=(32,32,3),data_format='channel_last')) #(None,30,30,16) [30 bcs no padding and stride=1]
model.add(MaxPooling2D((3,3))) #(None,10,10,16)
model.add(Flatten()) #(None,1600) (flatten is similar to linear)
model.add(Dense(64,activation='relu')) #(None,64)
model.add(Dense(10,activation='softmax')) #(None,10)
```
```
Model: "sequential_4"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_1 (Conv2D)            (None, 30, 30, 16)        448       
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 10, 10, 16)        0         
_________________________________________________________________
flatten_1 (Flatten)          (None, 1600)              0         
_________________________________________________________________
dense_5 (Dense)              (None, 64)                102464    
_________________________________________________________________
dense_6 (Dense)              (None, 10)                650       
=================================================================
Total params: 103,562
Trainable params: 103,562
Non-trainable params: 0
_________________________________________________________________

```


Pooling reduces size and provides translational invariance to input.
Stride reduces the size

**Weights**

The default values of the weights and biases in TensorFlow depend on the type of layers we are using.

For example, in a Dense layer, the biases are set to zero (zeros) by default, while the weights are set according to glorot_uniform, the Glorot uniform initialiser.

To customize weight initialize:
```
import tensorflow.keras.backend as K
def my_init(shape, dtype=None):
    return K.random_normal(shape, dtype=dtype)

model.add(Dense(64, kernel_initializer=my_init))
```

**Compile Method**

```
model.compile(optimizer='sgd', #'adam','rmsprop' 
              loss='binary_crossentropy', #'mean_squared_error','categorical_crossentropy'
              metrics=['accuracy','mae'])
```

or 

```
model.compile(optimizer=tf.keras.optimizers.SGD(),
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), #linear activation in last layer + sigmoid effect 
              metrics=[tf.keras.metrics.BinaryAccuracy(threshold=0.7)])
```

```
print(model.loss)
print(model.optimizer.lr) #to print learning rate
```

**(Sparse) Top  𝑘 -categorical accuracy**

In top  𝑘 -categorical accuracy, instead of computing how often the model correctly predicts the label of a training example, the metric computes how often the model has  𝑦𝑡𝑟𝑢𝑒  in the top  𝑘  of its predictions. By default,  𝑘=5 .

As before, the main difference between top  𝑘 -categorical accuracy and its sparse version is that the former assumes  𝑦𝑡𝑟𝑢𝑒  is a one-hot encoded vector, whereas the sparse version assumes  𝑦𝑡𝑟𝑢𝑒  is an integer.
















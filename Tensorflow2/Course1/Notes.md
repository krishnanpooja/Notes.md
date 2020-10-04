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


**Fit Method**

```
history = model.fit(x_train,y_train,epoch=10,batch_size=16)
#x_train = numpy array of (num_samples,num_features)
#y_train = numpy array of (num_samples,num_classes) #one-hot 
#history contains the history of loss....
```
if the classes are basically numbers (i.e, y_train =(num_samples)) then use sparse_categorical_crossentropy if y_train is one_hot representation then use categorical_crossentropy. 


**Evaluate method**

```
loss,accuracy,mae = model.evalute(x_test,y_test)
pred = model.predict(x_sample)
```
add dummy channel to sample  :  x[...,np.newaxis]


## Week 3 - Validation, regularisation and callbacks

Overfit when the training performance increases and validation performance decreases

```
Dense(64,kernel_regularizer=tf.keras.regularizers.l2(0.001),#weight_decay =0.01. other regularizers =l1, l1_l2
               bias_regularizer= tf.keras.regularizers.l1(0.005) ) 
```

**Batch Normalization**

```
BatchNormalization(),  # <- Batch normalisation layer
```
There are some parameters and hyperparameters associated with batch normalisation.

The hyperparameter momentum is the weighting given to the previous running mean when re-computing it with an extra minibatch. By default, it is set to 0.99.

The hyperparameter  𝜖  is used for numeric stability when performing the normalisation over the minibatch. By default it is set to 0.001.

The parameters  𝛽  and  𝛾  are used to implement an affine transformation after normalisation. By default,  𝛽  is an all-zeros vector, and  𝛾  is an all-ones vector.

```
# Add a customised batch normalisation layer

model.add(tf.keras.layers.BatchNormalization(
    momentum=0.95, 
    epsilon=0.005,
    axis = -1,
    beta_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05), 
    gamma_initializer=tf.keras.initializers.Constant(value=0.9)
))
```

**Callbacks**

```
from tensorflow.keras.callbacks import Callback

class my_callback(Callback):
   def on_train_begin(self,logs=None):
       #To do something at the start of the training
   
   def on_train_begin(self,batch,logs=None):
       #To do something at the start of the batch iteration

model.fit(X,y,epochs=5,callbacks=[my_callback()])
```

**EarlyStopping**

```
from tensorflow.keras.callbacks import EarlyStopping

earlyStopping = EarlyStopping(monitor='val_loss',patience =5 ,min_delta=0.01,mode=min) 
#patience = some epochs, training terminates if there is no improvement for 5 epochs
#min_delta sets a diff that is tolerated for val_loss. If bad then the patience is ++
#mode suggests that we are looking for the validation to minimize

model.fit(x,y,validation_split=0.2,epochs=100,callbacks=[earlyStopping])
```

**Logger**

```
tf.keras.callbacks.CSVLogger(filename, separator=',', append=False)
```
This callback streams the results from each epoch into a CSV file. The first line of the CSV file will be the names of pieces of information recorded on each subsequent line, beginning with the epoch and loss value. The values of metrics at the end of each epoch will also be recorded.

**Lambda callbacks**

```
tf.keras.callbacks.LambdaCallback(
        on_epoch_begin=None, on_epoch_end=None, 
        on_batch_begin=None, on_batch_end=None, 
        on_train_begin=None, on_train_end=None)
```
Lambda callbacks are used to quickly define simple custom callbacks with the use of lambda functions.

Each of the functions require some positional arguments.

on_epoch_begin and on_epoch_end expect two arguments: epoch and logs,
on_batch_begin and on_batch_end expect two arguments: batch and logs and
on_train_begin and on_train_end expect one argument: logs.

**Reduce learning rate on plateau**

```
tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', 
            factor=0.1, 
            patience=10, 
            verbose=0, 
            mode='auto', 
            min_delta=0.0001, 
            cooldown=0, 
            min_lr=0)
```
The ReduceLROnPlateau callback allows reduction of the learning rate when a metric has stopped improving. The arguments are similar to those used in the EarlyStopping callback.

### Week 4 - Saving and loading model weights

```
from tensorflow.keras.callbacks import ModelCheckpoint
checkpoint = ModelCheckpoint('my_model.{epoch}.{batch}',save_weights_only=True,save_freq=1000,save_best_only=True,monitor='val_loss',mode='min') #save_freq is the # of samples that the model has seen.
```
or
```
model.save_weights('my_model')
```

checkpoint
This file is by far the smallest, at only 87 bytes. It's actually so small that we can just look at it directly. It's a human readable file with the following text:

model_checkpoint_path: "checkpoint"
all_model_checkpoint_paths: "checkpoint"
This is metadata that indicates where the actual model data is stored.

checkpoint.index
This file tells TensorFlow which weights are stored where. When running models on distributed systems, there may be different shards, meaning the full model may have to be recomposed from multiple sources. In the last notebook, you created a single model on a single machine, so there is only one shard and all weights are stored in the same place.

checkpoint.data-00000-of-00001
This file contains the actual weights from the model. It is by far the largest of the 3 files. Recall that the model you trained had around 14000 parameters, meaning this file is roughly 12 bytes per saved weight.
to load the weights:


**Load weights**
```
model.load_weights('my_model')
```

**When you save the whole model:** 
Architecture + weights
. Assets folder - Contans files for tensorboard
. Variables folder -Contains the saved weights of the model
. saved_model.pb - Tensorflow graph

**To load the model**
```
from tensorflow.keras.models imports load_model

new_model = load_model('my_model') #filepath
new_model.summary
```

**To load pretrained model**
```
from tensorflow.keras.applications.resnet50 import ResNet50

model = Resnet50(weights='imagenet',include_top=True) #complete model including the classification layer

**TensorFlow hub**

Seperate library
```
import tensorflow_hub as hub

module = load_model(path)
model = Sequential(hub.KerasLayer(module))
model.build(input_shape=[None,160,160,3])
model.summary()
```
Parameters are non-trainable i.e, only forward pass possible

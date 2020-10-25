## Course 2 

### Week 1 - Introduction

**Functional API**

 Used when model has 
- multiple input and output
- model has conditional variables
- complicated non linear topology

```
from tensorflow.keras.layers import Input, Dense #no seperate Input layer in sequential API
from tensorflow.keras.models import Models #instead of regular import Sequential

inputs=Input(shape=(32,1))
h = Conv1D(16,5,activation='relu')(inputs)
h = AveragePooling1D(3)(h)
h = Flatten()(h)
outputs = Dense(20,activation='sigmoid')(h)

model = Model(inputs = input, outputs = outputs)
```

**Multiple Input and output**

```
from tensorflow.keras.layers import Input, Dense #no seperate Input layer in sequential API
from tensorflow.keras.models import Models #instead of regular import Sequential

inputs=Input(shape=(32,1))
h = Conv1D(16,5,activation='relu')(inputs)
h = AveragePooling1D(3)(h)
h = Flatten()(h)
aux_inputs=Input(shape=(12,))
h = Concatenate()([h,aux_inputs]) #concatenated along last dimension
outputs = Dense(20,activation='sigmoid')(h)
aux_outputs = Dense(1,activation='linear')(h)

model = Model(inputs = [input,aux_inputs], outputs = [outputs,aux_outputs])
model.compile(loss=['binary_crossentropy','mse'],loss_weights=[1,0.4], metrics=['acuracy'])
history = model.fit([X_train, X_aux],[y_train,y_aux], validation_split = 0.2 , epochs = 20)
#or
model.compile(loss={‘outputs’: ‘mse’, ‘aux_outputs’: ‘categorical_crossentropy’})
model.fit({‘main_inputs’: X_train1, ‘aux_inputs’: X_train2}, {‘main_outputs’: y_train1, ‘aux_outputs’: y_train2})
```
Can help build complex graph

**Variables**

Object whose values can be changes like weights, learning rate.(model parameters)
For a Dense layer, you have dense/kernel and dense/bias variables automatically declared when you call the Dense function.

```
my_var = tf.Variable([-1,2],dtype = tf.float32,name= 'my_var')
my_var.assign([3.5,-1.])
print(my_var)

x = my_var.numpy()
```

**Tensors**

```
my_var = tf.Variable([-1,2],dtype = tf.float32,name= 'my_var')
h = my_var + [5,4]
print(h) # prints a tensor
x = tf.ones((2,1))
y = tf.zeros((2,1))
```

**Accessing model layers**

```
print(model.layers[1]) #returns the second layer in the model
print(model.layers[1].weights)
print(model.layers[1].kernel)
print(model.layers[1].bias)
print(model.layers[1].get_weights()) 
print(model.get_layer('layer_name').bias)
```

**Transfer Learning**
```
print(model.get_layer('layer_name').input)
print(model.get_layer('layer_name').output) #every layer has input and output so does the model


model2 = Model(inputs = model.input,outputs = model.get_layer('flatten_layer').output) #if you want to skip the final sigmoid layer and create a new model

model3 = Sequential([model2,
          Dense(10,activation='softmax'])    
          
```
**Freezing layers**

Freeze conv layer by setting the trainable to False
```
h = Conv2D(16,3,actvation='relu',name='cov2d',trainable = False)(inputs)
```
or
```
model.get_layer('conv2d').trainable = False
```
or
freeze the whole model
```
model = load_weight('trained_model.pt')
model.trainable = False 
flatten_output = model.get_layer('flatten_layer').output
new_output = Dense(5,activation='softmax',name=new_softmax_layer')(flatten_output)
new_model = Model(inputs = model.input, outputs=new_outputs)
new_model.compile(loss='sparse_categorical_entropy')
new_model.fit(X_train,y_train.epochs=10)
```
**Device Placement**

```
tf.config.list_physical_devices()

# Get the GPU device name
tf.test.gpu_device_name()

if tf.config.experimental.list_physical_devices("GPU"):
    print("On GPU:")
    with tf.device("GPU:0"): 
        x = tf.random.uniform([1000, 1000])
        assert x.device.endswith("GPU:0")
        time_matadd(x)
        time_matmul(x)
```


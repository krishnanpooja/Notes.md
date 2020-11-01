## Course 2 

### Week 2 - Functional API, tf.Data

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

### Week 2 - Data Pipeline

**Datasets**

```
from tensorflow.keras.datasets import mnist, cifar10, imdb
(x_train,y_train) ,(x_test,y_test) = mnist.load_data()
(x_train,y_train) ,(x_test,y_test) = imdb.load_data(num_words=1000, max_len=100)
```

**Generators**

Used when dataset doesn't fit into the memory. Use 'yield' to do this.

```
datagen = get_data(32)
x,y = next(datagen)

model.fit_generator(datagen, steps_per_epoch = 1000, epochs = 10)
```
After 1000 iterations its counts it as one epoch.
or

```
for - in range(10000):
    x_train, y_train = next(datagen)
    model.train_on_batch(x_train,y_train)
```

```
model.evaluate_generator(datagen_eval, steps=100)
model.predict_generator(datagen_test, steps=100)
```

**Image Data Augmentation**

```
from tensorflow.keras.preprocessing.image import ImageDataGenerator

image_data_gen = ImageDataGenerator(rescale = 1/255., horizontal_flip = True , height_shoft_range = 0.2 , fill_mode = 'nearest')
image_data_gen.fit(x_train)
train_datagen = image_data_gen.flow(x_train, y_train, batch_size = 16)
model.fit_generator(train_datagen, steps_per_epoch = 10 )
```

```
datagen = ImageDataGenerator(rescale = (1/255.0))
train_gen = datagen.flow_from_directory(train_path, batch_size = 64, classes = classes, target_size = (16,16))
train_steps_per_epoch = train.generator.n // train_generator.batch_size)
```

**Time Series Generator**

```
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
timeseries_gen = TimeseriesGenerator(dummy_data, dummy_targets, length)
timeseries_iterator = iter(timeseries_gen)
```
```
# Create a TimeseriesGenerator object with a stride of 2

timeseries_gen = TimeseriesGenerator(dummy_data, dummy_targets, length=3, stride=2, batch_size=1)
```
Above we have specified the length as 3, and the stride as 2. This means that we will generate sequences starting with the first sample being (1, 2, 3) to predict the target 40. Subsequent samples will each skip 2 timesteps since the stride is 2, meaning the next sample will will be (3, 4, 5) to predict 60, and the one after that will be (5, 6, 7) to predict 80.

```
# Create a reversed TimeseriesGenerator object

timeseries_gen = TimeseriesGenerator(dummy_data, dummy_targets, length=3, stride=1, batch_size=1, reverse=True)
timeseries_iterator = iter(timeseries_gen)

while True:
    try:
        print(next(timeseries_iterator))
    except StopIteration:
        break
#(array([[3, 2, 1]]), array([40]))
#(array([[4, 3, 2]]), array([50]))
#(array([[5, 4, 3]]), array([60]))
#(array([[6, 5, 4]]), array([70]))
#(array([[7, 6, 5]]), array([80]))
#(array([[8, 7, 6]]), array([90]))
#(array([[9, 8, 7]]), array([100]))
```
Load Audio File

```
from scipy.io.wavfile import read, write

rate, song = read("data/055 - Angels In Amplifiers - I'm Alright/mixture.wav")
print("rate:", rate)
song = np.array(song)
print("song.shape:", song.shape)
```

**tf.data**

```
dataset = tf.data.dataset.from_tensor_slices([1,2,3,4,5])
for elem in dataset:
   print(elem.numpy())
print(dataset.element_spec)

for elem in dataset.take(2):
    print(elem)
```

```
(x_train,y_train) ,(x_test,y_test) = cifar10.load_data()
dataset = tf.data.dataset.from_tensor_slices((x_train,y_train))
```

```
image_data_gen = ImageDataGenerator(rescale = 1/255., horizontal_flip = True , height_shoft_range = 0.2 , fill_mode = 'nearest')
dataset = tf.data.dataset.from_generator(image_data_gen.flow,args=[x_train,y_train]
                                         output_types = (tf.float32,tf.int32),
                                         output_shapes = ([32,32,32,3],[32,1])) #flow method sets up batch size of 32 by default
                                         
```

Different dataset shape:

```
dataset_zipped = tf.data.Dataset.zip([dataset1,dataset2])
```

TextLineDataset:

 ```
 test_files = sorted([f.path for f in os.scandir(data/shakespeare)])
 shakespeare_dataset = tf.data.TextLineDataset(text_files)
 first_5_lines_dataset = iter(shakespeare_dataset.take(5))
 lines = [line for line in first_5_lines_dataset]
 for line in lines:
      print(line)
 ```
 
 Interleave:
 If you want the first line of all the text files followed by the second line of the files.
 ```
 text_file_dataset = tf.data.Dataset.from_tensor_slices(text_files)
 files = [file for file in test_file_dataset]
 interleave_dataset = text_file_dataset.interleave(tf.data.TextLineDataset, cycle_length = 9)
 ```
The main distinction between the from_tensor_slices function and the from_tensors function is that the from_tensor_slices method will interpret the first dimension of the input data as the number of elements in the dataset, whereas the from_tensors method always results in a Dataset with a single element, containing the Tensor or tuple of Tensors passed. (The zeroth dimension need not be same for from_tensor)
 To use pandas dataframe. First convert the dataframe into dict() and then pass it to tf.data.Dataset()
 
 **Training with datasets**
 
 ```
 def rescale(img,label):
     return img/255,label
     
  def label_filter(img,label): #removes all the images with label 9
      return tf.squeeze(label)!=9
      
 dataset = tf.data.dataset.from_tensor_slices((x_train,y_train))
 
 dataset = dataset.map(rescale)
 dataset = dataset.filter(label_filter)
 
 dataset = dataset.batch(16,drop_reminder=True) #(16,32,32,3)
 dataset = dataset.repeat() # data will repeat infintely
 
 history = model.fit(dataset,steps_per_epoch=x_train.shape[0]//16,epochs =10)
 ```
 
 

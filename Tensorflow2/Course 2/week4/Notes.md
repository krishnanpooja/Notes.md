## Course 2 

### Week 4 - Fully customizable layers and models

**Model Subclassing API**

```
class MyModel(Model):
    def __init__(self, num_classes, **kwargs):
      super(MyModel, self).__init__(**kwargs)
      self.dense1 = Dense(16, activation='sigmoid')
      self.dropout = Dropout(0.5)
      self.dense2 = Dense(num_classes, activation='softmax')
      
     
    def call(self, inputs, training=False):
      h = self.dense1(inputs)
      h = self.dropout(h, training=training)
      return self.dense2(h)

my_model = MyModel(10, name='my_model')
```
**Custom Layers**

```
from tensorflow.keras.layers import Layer

class LinearMap(Layer):
  def __init__(self, input_dim, units):
    super(LinearMap, self).__init__()
    w_init = tf.random_normal_initializer()
    self.w = tf.Variable(initial_value = w_init(shape=(input_dim, units)))
    
   def call(self, inputs):
    return tf.matmul(inputs, self.w)

linear_layer = LinearMap(3,2)
print(linear_layer) # w_init values
```
**Automatic Differentiation**
```
import tensorflow as tf

x = tf.constant([-1, 0, 1], dtype=tf.float32)

with tf.GradientTape() as tape:
    tape.watch(x)
    y = tf.math.exp(x)
    z = 2 * tf.reduce_sum(y)
    dz_dx = tape.gradient(z, x)
```

Train the model
```
learning_rate = 0.05
steps = 25

for i in range(25):
    with tf.GradientTape() as tape:
        predictions  = linear_regression(x_train) # model function that computes y=mx+c
        loss = SquaredError(predictions, y_train)
        
    gradients = tape.gradient(loss, linear_regression.trainable_variables)
    
    linear_regression.m.assign_sub(learning_rate * gradients[0])
    linear_regression.b.assign_sub(learning_rate * gradients[1])
    
    print("Steps %d, Loss:%f".format(i,loss.numpy()))
```
    

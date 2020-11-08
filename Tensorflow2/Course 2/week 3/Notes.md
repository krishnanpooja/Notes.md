## Course 2 

### Week 3 - Sequential Data

**Stacked RNN**

```
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input,Masking, LSTM, Dense


inputs = Input(shape(None,10)) #(None, None,10)
h = Masking(mask_value=0)(inputs)  #(None, None,10)
h = LSTM(32, return_sequences = True)(h)  #(None, None,32)
h = LSTM(64)(h)   #(None, 64) Output of the final layer
outputs = Dense(5, activation='softmax')(h)  #(None, 5)

model = Model(inputs,outputs)
```

**Bidirectional**

```
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input,Masking, LSTM, Dense, Bidirectional


inputs = Input(shape(None,10)) #(None, None,10)
h = Masking(mask_value=0)(inputs)  #(None, None,10)
h = Bidirectional(LSTM(32, return_sequences=True))(h)  #(None, None,64)
h = Bidirectional(LSTM(64))(h)   #(None, 64)  #(None, 128)
outputs = Dense(5, activation='softmax')(h)  #(None, 5)

model = Model(inputs,outputs)
```
merge_mode = True
```
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input,Masking, LSTM, Dense, Bidirectional


inputs = Input(shape(None,10)) #(None, None,10)
h = Masking(mask_value=0)(inputs)  #(None, None,10)
h = Bidirectional(LSTM(32, return_sequences=True), merge_mode='sum')(h)  #(None, None,64)
h = Bidirectional(LSTM(64))(h)   #(None, 64)  #(None, 128)
outputs = Dense(5, activation='softmax')(h)  #(None, 5)

model = Model(inputs,outputs)
```
merge_mode deals with how the output layers are combined. It could take value like sum, null, concat.


```

max_index_val = max(imbd_word_index.value())
model = tf.keras.Sequential([
                              tf.keras.layers.Embedding(input_dim=max_index_val+1, output_dim = embedding_dim. mask_zero=True),
                              tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=32, return_sequences=True),merge_mode = 'sum',
                                                             backward_layer = tf.keras.layer.GRU(units=8, go_backwards=True)),
                              tf.keras.layers.Dense(units=1, activation='sigmoid')
                          ])                 

```

 the internal state of the stateful RNN after processing each batch is the same as it was earlier when we processed the entire sequence at once.

This property can be used when training stateful RNNs, if we ensure that each example in a batch is a continuation of the same sequence as the corresponding example in the previous batch.

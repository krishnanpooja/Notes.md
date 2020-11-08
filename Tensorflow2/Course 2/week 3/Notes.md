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

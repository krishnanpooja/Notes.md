__Learning Objectives__

1. Recognize the cases where Neural Architecture Search is the appropriate tool to find the right model architecture.
2. Distinguish between trainable parameters and hyperparameters
3. Judge when manual parameter search does not scale well
4. Identify search spaces and summarize the best strategies to navigate this space to find optimal architectures.
5. Differentiate the advantages and disadvantages AutoML for different use cases
6. Carry out different metric calculations to assess AutoML efficacy
7. Identify some cloud AutoML offerings and recognize their strengths and weaknesses

## Neural Architecture Search
Neural architecture search, or NAS, it's a technique for automating the design of neural networks. 
- Models found by NAS are often on par with or outperform hand designed architectures for many types of problem.
- The goal of NAS is to find the optimal architecture.
- The Keras team has released one of the best, Keras Tuner, which is a library to easily perform hyperparameter tuning with TensorFlow 2.0.

  ### Keras Auto Tuner
 I'm using the Hyperband strategy. It also supports random search and Bayesian optimization and Sklearn strategies. You can learn more about these at the Keras tuner site at Keras-team.github.io/kerastuner.
  
 ```
# Install Keras Tuner
!pip install -q -U keras-tuner 
```

```
import keras_tuner as kt
def model_builder(hp):
  '''
  Builds the model and sets up the hyperparameters to tune.

  Args:
    hp - Keras tuner object

  Returns:
    model with hyperparameters to tune
  '''

  # Initialize the Sequential API and start stacking the layers
  model = keras.Sequential()
  model.add(keras.layers.Flatten(input_shape=(28, 28)))

  # Tune the number of units in the first Dense layer
  # Choose an optimal value between 32-512
  hp_units = hp.Int('units', min_value=32, max_value=512, step=32)
  model.add(keras.layers.Dense(units=hp_units, activation='relu', name='tuned_dense_1'))

  # Add next layers
  model.add(keras.layers.Dropout(0.2))
  model.add(keras.layers.Dense(10, activation='softmax'))

  # Tune the learning rate for the optimizer
  # Choose an optimal value from 0.01, 0.001, or 0.0001
  hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

  model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
                loss=keras.losses.SparseCategoricalCrossentropy(),
                metrics=['accuracy'])

  return model

# Instantiate the tuner
tuner = kt.Hyperband(model_builder,
                     objective='val_accuracy',
                     max_epochs=10,
                     factor=3,
                     directory='kt_dir',
                     project_name='kt_hyperband')

# Perform hypertuning
tuner.search(img_train, label_train, epochs=NUM_EPOCHS, validation_split=0.2, callbacks=[stop_early])

# Get the optimal hyperparameters from the results
best_hps=tuner.get_best_hyperparameters()[0]

# Build the model with the optimal hyperparameters
h_model = tuner.hypermodel.build(best_hps)
h_model.summary()
```

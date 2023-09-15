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

## AutoML (Automated Machine Learning)

AutoML automates the development of ML models
- NAS is a subfield of AutoML
- 
3 main parts of NAS:-
1. Search Space - Architecture is picked from this space 
2. Search Strategy - how we explore the search space
3. Performance Estimation Strategy- selected architecture is passed here which returns estimated performance


#### Search Spaces
1. Macro - Node is a layer in neural netwrok like pooling layer Contains induvidual layers and connection types like a chained structure or complex search space
2. Micro - Cell based. Each cell is a samll network.
   Types of cells:
   1. Normal cell - on top
   2. reduction cell - below
   These cells are stacked to produce the final network. This search space is said to provide significant performance advantages.
   

#### Search Strategies
For smaller search spaces:
1. Grid search - Search everything in the grid
2. Random search

1. Bayesian Optimization -It assumes that a specific probability distribution, which is typically a Gaussian distribution is underlying the performance of model architectures. So you use observations from tested architectures to constrain the probability distribution and guide the selection of the next option.
2. Evolutionary Method-  First, an initial population of n different model architecture is randomly generated, the performance of each individual.Then the X highest performers are selected as parents for a new generation. This new generation of architectures might be copies of the respective parents with induced random alterations or mutations. Or they might arise from combinations of the parents, the performance of the offspring is assessed. Again using the performance estimation strategy. The list of possible mutations can include operations like adding or removing a layer. Adding or removing a connection, changing the size of a layer or changing another hyper parameter. Y architectures are selected to be removed from the population. This might be the Y worst performers, the Y oldest individuals in the population. Or a selection of individuals based on a combination of these parameters. The offspring replaces the removed architectures and the process is restarted with this new population
   
 <img width="906" alt="image" src="https://github.com/krishnanpooja/Notes.md/assets/8016149/dfb46832-a4f9-43a3-8630-025c0dd2c96e">

3. Reinforcement - Agents goal is to maximize the result.
     - The available options are selected from the search space
     - performance estimation determines the reward
       
 A neural network can also be specified by a variable length string where the elements of the string specify individual network layers.
That enables us to use a recurrent neural network or RNN to generate that string, as we might do for an NLP model. The RNN that generates the string is referred to as the controller, after training the network referred to as the child network on real data. We can measure the accuracy on the validation set, the accuracy determines the reinforcement learning reward in this case. Based on the accuracy, we can compute the policy gradient to update the controller RNN. In the next iteration, the controller will have learned to give higher probabilities to architectures that result in higher accuracies during training. 


#### Measuring AutoML Efficacy
1. Lower Fidelity Estimates
   - Reduces the training time by reframing the problem by making it easier to solve.
   - this is done by using data subset, low resolution images, fewer filters and cells.
   - This reduces the cost but underestimates the performance

2. Learning Curve extrapolation
  - Extrapolates based on initial learning
  - requires predicting learning curve reliably
  - helps remove poor performers

3. Weight Inheritance/ Network Morphisms
    - Initialize weights of new architectures based on prev trained architectures
    - similar to transfer learning
    - Underlying function remains unchanged
      - New network inherits knowledge from parent netwrok
      - Advantages:
      -  computational speed up: a few days of GPU
      -  Network size not inherently bounded
  
### Cloud based AutoML
#### Amazon SageMaker Autopilot
<img width="896" alt="image" src="https://github.com/krishnanpooja/Notes.md/assets/8016149/e6429fcb-c5fd-4029-8d91-10a94bcb4490">

Advantages
1. Quick Iterations
2. high quality models
3. Performance ranked 
4. selected features can be viewed
5. notebooks provide reproducabilty

#### Microsoft Azure AutoML 
<img width="917" alt="image" src="https://github.com/krishnanpooja/Notes.md/assets/8016149/8220db4d-8c1f-49b4-a586-a261ac88d582">

Advanategs:
1. Quick customization of model and control settings
2. Automated feature engineering
3. data visualization
4. intelligent stopping
5. experiment summaries and metric visualizations
6. model interpretabilty
7. pattern discovery


#### Google Cloud AutoML
Products:
1. Sight - AutoML Vision (images)- classification and edge image classification, object detection and edge object detection, AutoML Video Intelligence (Video)
2. Language - AutoML Natural Language (reveal structure and meaning of text), AutoML Translation
3. Structures data  AutoML Tables

   


### Explainable AI
- Why your model did what it did
- __Explainable Artificial Intelligence (XAI)__
    - its needed bcs
        1. models with high sensitvity which can generate widly wrong results
        2. attacks
        3. fairness
        4. reputation and branding
        5. legal and reglatory concerns
        6. customers and other stakeholders may question or challenge model decisions

#### Model Interpretabilty 
- interpretability methods can be grouped based on whether they're intrinsic or Post-Hoc. They could also be model-specific or model agnostic. They can also be grouped according to whether they are local or global

##### Intrinsic or Post-hoc
- __Intrinsic__ - The classic examples of this are linear models and tree-based models. Intrinsically interpretable model provides a higher degree of certainty as to why it generated a particular result
-  __Post-hoc__ - They tend to treat all models the same and are applied after training to try to examine particular results to understand what caused the model to generate them. There are some methods, however, especially for convolutional networks, that do inspect the layers within the network to try to understand how results were generated.

##### Model specific or Model Agnostic
- Model-specific methods are limited to specific model types. For example, the interpretation of regression weights and linear models is model-specific.
- By definition, the techniques for interpreting intrinsically interpretable models are model-specific, but model-specific methods are not limited to intrinsically interpretable models.
-  Model agnostic methods are not specific to any particular model. They can be applied to any model after it's trained.
-  Essentially, they are Post-hoc methods. These methods do not have access to the internals of the model such as weights and parameters, etc.

##### Local or global
- __local__ - A local interpretability method explains an individual prediction. For example, feature attribution in the prediction of a single example in the dataset.
- __global__  - Global methods explain the entire model behavior. For example, if the method creates a summary of feature attributions for predictions on the entire test set, then it can be considered global. 

### Intrinsically Interpretable Model
- working of the model is transparent enough to explain its working

#### TF Lattice
- Overlaps a grid onto the feature space and learns values for the output at the vertices of the grid
- linearly interpolates from the lattice values surrounding a point
- __TF lattice__ enables you to inject domain knowledge into the learning process through common-sence or policy-driven shape constraints
- Advamtage:- accuracy as per neural netwrok and interpretable unlike NN
- disadvantage:- high dimensions can complicate things


### Model Agnostic Methods
Model agnostic methods separate the explanations from the model. These methods can be applied to any model after it's been trained. For example, they can be applied to linear regression or decision trees or even black box models like neural networks.

#### Permutation Feature Importance
- Feature is important if shuffling its values increases model error
- Feature is __unimportant__ if shuffling leaves model error unchanged

  <img width="927" alt="image" src="https://github.com/krishnanpooja/Notes.md/assets/8016149/226d4699-e78a-46d2-a1d1-413852ea600a">

#### Shapley Value
- how important is each player to the overall cooperation and what payoff can he or she reasonably expect? The Shapley value provides one possible answer to this question.The players are the features of the data set, and we're using the Shapley value to determine how much each feature contributes to the results. 
- Advantages:
   1. Based on solid theoretical foundation
   2. value is fairly distriuted among all features
   3. enables contrastive explanations

- Disadvantages:
   1. Computationally expensive
   2. can be Misinterpreted
   3. always uses all features
   4. can't test for different input values
   5. doesn't work well when features are correleated

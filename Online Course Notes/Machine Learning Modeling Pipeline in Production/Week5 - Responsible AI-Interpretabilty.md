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

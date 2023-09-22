### Curse of Dimensionality

<img width="680" alt="image" src="https://github.com/krishnanpooja/Notes.md/assets/8016149/5d2a3f38-0f49-446d-ae4a-06538ad5517e">

<img width="432" alt="image" src="https://github.com/krishnanpooja/Notes.md/assets/8016149/64644879-718a-4a04-8e9e-c0bc3e27653b">

Regardless of which modeling approach you're using, increasing dimensionality has another problem especially for classification which is known as the Hughes effect. This is a phenomenon that demonstrates the improvement in classification performance as the number of features increases until we reach an optimum where we have enough features. Adding more features while keeping the training set the same size will degrade the classifiers performance. We saw this earlier in our graph. In classification, the goal is to find a function that discriminates between two or more classes. You can do this by searching for hyperplanes in space that separate these categories. The more dimensions you have, the easier it is to find a hyperplane during training, but at the same time the harder it is to match that performance when generalizing to unseen data. And the less training data you have, the less sure you are that you identify the dimensions that matter for discriminating between categories.

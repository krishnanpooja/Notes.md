### Curse of Dimensionality

<img width="680" alt="image" src="https://github.com/krishnanpooja/Notes.md/assets/8016149/5d2a3f38-0f49-446d-ae4a-06538ad5517e">

<img width="432" alt="image" src="https://github.com/krishnanpooja/Notes.md/assets/8016149/64644879-718a-4a04-8e9e-c0bc3e27653b">

Regardless of which modeling approach you're using, increasing dimensionality has another problem especially for classification which is known as the Hughes effect. This is a phenomenon that demonstrates the improvement in classification performance as the number of features increases until we reach an optimum where we have enough features. Adding more features while keeping the training set the same size will degrade the classifiers performance. We saw this earlier in our graph. In classification, the goal is to find a function that discriminates between two or more classes. You can do this by searching for hyperplanes in space that separate these categories. The more dimensions you have, the easier it is to find a hyperplane during training, but at the same time the harder it is to match that performance when generalizing to unseen data. And the less training data you have, the less sure you are that you identify the dimensions that matter for discriminating between categories.


#### Algorithmic Dimensionality Reduction Techniques

<img width="853" alt="image" src="https://github.com/krishnanpooja/Notes.md/assets/8016149/475a5fe6-3370-403b-ac58-040f7351737c">

#### Principal Component Analysis
- This is an unsupervised algorithm that creates linear combinations of the original features.
- The goal of PCA is that it tries to find a lower dimensional surface onto which to project the data so as to minimize the squared projection error. In other words, to minimize the square of the distance between each point and the location of where it gets projected. The result will be to maximize the variance of the projections.
- The first principal component is the projection direction that maximizes the variance of the projected data. The second principal component is the projection direction that is orthogonal to the first principal component and maximizes the remaining variance of the projected data.
- This reconstruction will of course, have some amount of error, but this is often negligible and acceptable given the other benefits of dimensionality reduction
- The factor loadings are the unstandardized values of the eigenvectors. We can interpret the loadings as the covariances or correlations. Scikit-learn has an implementation of PCA which includes both fit and transform methods, just like the standard scalar operation, as well as a fit transform method which combines both fit and transform.
- The fit method learns how to shift and rotate the samples, but it doesn't actually change them. The transform method, on the other hand, applies the transformation the fit learned.

<img width="900" alt="image" src="https://github.com/krishnanpooja/Notes.md/assets/8016149/acb74756-7842-4248-9491-f100bec87f2f">

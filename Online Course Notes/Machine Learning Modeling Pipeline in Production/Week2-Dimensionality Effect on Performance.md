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

#### More Dimensionality Reduction Techniques
##### SVD (Singular value decomposition)
- unsupervised approaches
- Matrices can be seen as linear transformations in space. PCA, which we discussed previously relies on eigen-decomposition, which can only be done for square matrices, of course, you don't always have square matrices.
- In TF-IDF, for example, high frequency of terms may not really be fruitful, in some cases, rare words contribute more. In general, the importance of words increases if the number of occurrences of these words within the same document also increases. On the other hand, the importance will be decreased for words which occur frequently in the corpus. The challenges that the resulting matrix is very sparse and not square. To decompose these types of matrices, which can't be decomposed with eigendecomposition, we can use techniques such as singular value decomposition or SVD.

##### Indepndent Component Analysis (ICA)
- PCAs and ICAs's significant difference is that PCA looks for uncorrelated factors, while ICA looks for independent factors.
- If two factors are uncorrelated, it means that there is no linear relation between them. If they're independent, it means that they are not dependent on other variables
- Independent component analysis separates a multivariant signal into additive components that are maximally independent.
- ICA is not used for reducing dimensionality but for separating superimposed signals. Since the model does not include a noise term, for the model to be correct, whitening must be applied. This can be done in various ways, including using one of the PCA variants. ICA further assumes that there exists independent signals, S, and a linear combination of signals, Y. The goal of ICA Is to recover the original signals, S, from Y, ICA assumes that the given variables are linear mixtures of some unknown latent variables. It also assumes that these latent variables are mutually independent. In other words, they're not dependent on other variables and hence they are called the independent components of the observed data.
- Compare PCA and ICA
    - Both are statistical transformations, that is PCA uses information extracted from second order statistics, while ICA goes up to higher order statistics
    - Both are used in various fields like blind source separation, feature extraction and also in neuroscience. ICA is an algorithm that finds directions in the feature space corresponding to projections which are highly non-Gaussian
    - Unlike PCA, these directions need not be orthogonal in the original feature space, but they are orthogonal in the whitened feature space, in which all directions correspond to the same variance. PCA, on the other hand, finds orthogonal directions in the raw feature space that corresponded directions accounting for maximum variance.
    - When it comes to the importance of components, PCA, considers some of them to be more important than others. ICA, on the other hand, considers all components to be equally important.
 
#### Non-Negative Matrix Factorization(NMF)
- NMF expresses samples as a combination of interpretable parts.
- NMF, like PCA, is a dimensionality reduction technique, in contrast to PCA, however, NMF models are interpretable
- This means NMF models are easier to understand and much easier for us to explain to others
- NMF can't be applied to every dataset however, it requires the sample features to be non-negative, so the values must be greater than or equal to zero.
- It has been observed that when carefully constrained, NMF can produce a parts-based representation of the dataset, resulting in interpretable models

### Quantization
__why Quantize__
1. Neural networks have many params and take up space
2. shrinking model file size
3. reduce computational resources
4. make models run faster
5. use less power with lowprecision

Quantization squeezes small range of floating point values into a fixed number of information buckets

__what part of models are affected?__
1. static values(paramters)
2. dynamic values(activations)
3. compuations(transformations)

__Trade-offs__
1. Optimizations impact model accuracy
2. in rare cases, models may actually gain accuracy
3. undefined effects on ML interpretabilty


__You can do quantization during training or after the model has been trained__
#### Post-training Quantization -quantize the weights from floating point numbers to integers in an efficient way
Post-training quantization is a conversion technique that can reduce model size while also improving CPU and hardware accelerator latency with little degradation in model accuracy.
You can quantize an already trained TensorFlow model when you convert it to TensorFlow Lite format using the TensorFlow Lite converter.

<img width="864" alt="image" src="https://github.com/krishnanpooja/Notes.md/assets/8016149/2811a2a6-9556-43ca-a3c0-78c0a2d1c5c3">

- With dynamic range quantization, during inference, the weights are converted from eight bits to floating point, and the activations are computed using floating point kernels. This conversion is done once and cached to reduce latency. This optimization provides latencies which are close to fully fixed point inference.
- Using dynamic range quantization, you can reduce the model size and/or latency, but this comes with a limitation as it requires inference to be done with floating point numbers. This may not always be ideal since some hardware accelerators only support integer operations, for example, Edge TPUs.
- Post-training integer quantization works by gathering calibration data, which it does by running inferences on a small set of inputs so as to determine the right scaling parameters needed to convert the model to an integer quantized model

<img width="871" alt="image" src="https://github.com/krishnanpooja/Notes.md/assets/8016149/2e740684-9d74-4613-b5de-3bc397f650ad">

 if the loss of accuracy is too great, consider using quantization aware training. However, doing so requires modifications during model training to add fake quantization nodes,

#### Pruning
- Another method to increase the efficiency of models is to remove parts of the model that did not contribute substantially to producing accurate results. This is referred to as pruning. 
- 1990, written by Yann LeCun, John S. Denker, and Sara A. Solla and was given the rather provocative title of 'Optimal Brain Damage'
- Pruning was mainly done by using magnitude as an approximation for saliency to determine less useful connections. The intuition being that smaller magnitude weights have a smaller effect in the output, and hence are less likely to have an impact in the model outcome if proven. It was a iterative pruning method.
- Pruning was mainly done by using magnitude as an approximation for saliency to determine less useful connections. The intuition being that smaller magnitude weights have a smaller effect in the output, and hence are less likely to have an impact in the model outcome if proven. It was a iterative pruning method.
- That over parameterized dense networks containing several sparse subnetworks with varying performances, and one of these subnetworks is the winning ticket, which outperforms all others. However, there were significant limitations to this method. For one, it does not perform well for larger-scale problems and architectures. 
<img width="717" alt="image" src="https://github.com/krishnanpooja/Notes.md/assets/8016149/8f9f7207-9312-4787-897b-00521da1392d">


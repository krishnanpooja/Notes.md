**How do we form a confidence interval?**	

The purpose of taking a random sample from a lot or population and computing a statistic, such as the mean from the data, is to approximate the mean of the population. How well the sample statistic estimates the underlying population value is always an issue. A confidence interval addresses this issue because it provides a range of values which is likely to contain the population parameter of interest.

**Confidence Level**

Confidence intervals are constructed at a confidence level, such as 95 %, selected by the user. What does this mean? It means that if the same population is sampled on numerous occasions and interval estimates are made on each occasion, the resulting intervals would bracket the true population parameter in approximately 95 % of the cases. A confidence stated at a 1−α level can be thought of as the inverse of a significance level, α.

**Oversampling and Undersampling**

The two main approaches to randomly resampling an imbalanced dataset are to delete examples from the majority class, called undersampling, and to duplicate examples from the minority class, called oversampling.
- SMOTE-synthetic dataset generation
- combination of under and over sampling

**ROC and AUC**

This type of graph is called a Receiver Operating Characteristic curve (or ROC curve.) It is a plot of the true positive rate against the false positive rate for different thresholds.
True positive rate = TP/(TP+FN) and False positive rate= FP/(FP+TN)

An ROC curve demonstrates several things:
It shows the tradeoff between sensitivity and specificity (any increase in sensitivity will be accompanied by a decrease in specificity).
The closer the curve follows the left-hand border and then the top border of the ROC space, the more accurate the test.
The closer the curve comes to the 45-degree diagonal of the ROC space, the less accurate the test.

Area under the curve(AUC) helps determine which curve is better.It tells how much model is capable of distinguishing between classes. Higher the AUC, better the model is at predicting 0s as 0s and 1s as 1s. 

metrics like precision is used when the like a new disease where there are less people affected

**Random Forest**

Random forest algorithm creates decision trees on data samples and then gets the prediction from each of them and finally selects the best solution by means of voting. It is an ensemble method which is better than a single decision tree because it reduces the over-fitting by averaging the result.

**Overfitting and Underfiting**

Overfitting occurs when a statistical model or machine learning algorithm captures the noise of the data.  Intuitively, overfitting occurs when the model or the algorithm fits the data too well.  Specifically, overfitting occurs if the model or algorithm shows low bias but high variance.  Overfitting is often a result of an excessively complicated model, and it can be prevented by fitting multiple models and using validation or cross-validation to compare their predictive accuracies on test data.

low bias and high variance. model tries to fit to training data. Training performance good but test time the performance decreases.The noise is very high.

Underfitting occurs when a statistical model or machine learning algorithm cannot capture the underlying trend of the data.  Intuitively, underfitting occurs when the model or the algorithm does not fit the data well enough.  Specifically, underfitting occurs if the model or algorithm shows low variance but high bias.  Underfitting is often a result of an excessively simple model.

high bias and low variance. Happens when we use linear model to fit non-linear data. or have very less quantity of data.

**Early Stoppping**

Point where the validation error increases and while training error is decreasing.

**Why is Bias Variance Tradeoff?**

If our model is too simple and has very few parameters then it may have high bias and low variance. On the other hand if our model has large number of parameters then it’s going to have high variance and low bias. So we need to find the right/good balance without overfitting and underfitting the data.
This tradeoff in complexity is why there is a tradeoff between bias and variance. An algorithm can’t be more complex and less complex at the same time.

**Difference between SVM and Logistic Regression**

SVM tries to finds the “best” margin (distance between the line and the support vectors) that separates the classes and this reduces the risk of error on the data, while logistic regression does not, instead it can have different decision boundaries with different weights that are near the optimal point.

SVM works well with unstructured and semi-structured data like text and images while logistic regression works with already identified independent variables.

SVM is based on geometrical properties of the data while logistic regression is based on statistical approaches.

The risk of overfitting is less in SVM, while Logistic regression is vulnerable to overfitting.

SVM works better with higher dimensional data unlike logistic regression which is good with low dimensional.

SVM uses few data points called support vectors unlike logistic regression which uses the entire training data.


Problems that can be solved using SVM
- Image classification
- Recognizing handwriting
- Cancer detection

Problems to apply logistic regression algorithm
- Cancer detection — can be used to detect if a patient have cancer (1) or not(0).
- Test score — predict if a student passed(1) or failed(0) a test.

**What is Support Vector Machine?**

The objective of the support vector machine algorithm is to find a hyperplane in an N-dimensional space(N — the number of features) that distinctly classifies the data points.
 
Our objective is to find a plane that has the maximum margin, i.e the maximum distance between data points of both classes.

Support vectors are data points that are closer to the hyperplane and influence the position and orientation of the hyperplane. Using these support vectors, we maximize the margin of the classifier. Deleting the support vectors will change the position of the hyperplane. These are the points that help us build our SVM.

In SVM, we take the output of the linear function and if that output is greater than 1, we identify it with one class and if the output is -1, we identify is with another class. Since the threshold values are changed to 1 and -1 in SVM, we obtain this reinforcement range of values([-1,1]) which acts as margin.

Loss: Hinge Loss. L = 1-y*f(x) 
The cost is 0 if the predicted value and the actual value are of the same sign. If they are not, we then calculate the loss value. We also add a regularization parameter the cost function

**L1 and L2 regularization**

Ridge:L2 regularization- It includes all (or none) of the features in the model. Thus, the major advantage of ridge regression is coefficient shrinkage and reducing model complexity.
Lasso: L1 regularization- Along with shrinking coefficients, lasso performs feature selection as well. (Remember the ‘selection‘ in the lasso full-form?) As we observed earlier, some of the coefficients become exactly zero, which is equivalent to the particular feature being excluded from the model.Lower Computational costs. Can be used for higher dimensional data.

**Bloom Filter**

Initialization: Create an array B initialize with 0. Use each object in set S, compute hash value and mark as 1
To check if spam: compute hashmap for 'x'. use 'k' hashmaps. If all the 'k' hashmap to 1 then not spam.
Marketing — predict if a customer will purchase a product(1) or not(0).

**Singular Value Decomposition: SVD**

 A= U(sigma)V_T
Since uᵢ and vᵢ have unit length, the most dominant factor in determining the significance of each term is the singular value σᵢ. We purposely sort σᵢ in the descending order. If the eigenvalues become too small, we can ignore the remaining terms (+ σᵢuᵢvᵢᵀ + …

**PCA and ICA**

Principal Component Analysis: PCA
Look for high variance to reduce the dimensionality of the data
PCA is a linear model in mapping m-dimensional input features to k-dimensional latent factors (k principal components). If we ignore the less significant terms, we remove the components that we care less but keep the principal directions with the highest variances (largest information).
Not all components are important it depends on eigen value
remove correlations but not higher order dependence

ICA Independent component analysis
Consider n signals at time t, Y=AS(t).You need to find value to retrieve S back i.e. A inverse
Removes correlation and high order dependence
All the components are equally important

**K-Means**

K-Means algorithm
1. initialize mu points 
2. compute the euclidean distance to the centroid. Assign data to the closest centroids
3. recompute the centroids (summation of data points in that cluster/no of data points)
4. repeat until convergence

**Generative and discriminative model**

Generative - models actual distribution of each class. P(y,x) - joint probability-- [learn stuff indepth]
- Naïve Bayes
- Bayesian networks
- Markov random fields
- Hidden Markov Models (HMM)

Discriminative - models the decision boundary between the classes. P(y|x) -- conditional probability--[look for differences]
- Logistic regression
- Scalar Vector Machine
- Traditional neural networks
- Nearest neighbour
- Conditional Random Fields (CRF)s

If data is highly correlated, we should expect many σᵢ values to be small and can be ignored.

**Type-I and Type-II error**

Type -I error- False positive 
Type -II error- False negatives

Precision = TP/(TP+FP)

Recall = TP/(TP+FN)

Type I error is a false positive, while Type II error is a false negative. Briefly stated, Type I error means claiming something has happened when it hasn’t, while Type II error means that you claim nothing is happening when in fact something is. A clever way to think about this is to think of Type I error as telling a man he is pregnant, while Type II error means you tell a pregnant woman she isn’t carrying a baby.

**SIFT**

Scale Invariant Feature Transform, is a feature detection algorithm in Computer Vision.
SIFT helps locate the local features in an image, commonly known as the ‘keypoints‘ of the image. These keypoints are scale & rotation invariant that can be used for various computer vision applications, like image matching, object detection, scene detection, etc.

We can also use the keypoints generated using SIFT as features for the image during model training. The major advantage of SIFT features, over edge features or hog features, is that they are not affected by the size or orientation of the image.

Gaussian Blurring technique to reduce the noise in an image

https://www.analyticsvidhya.com/blog/2019/10/detailed-guide-powerful-sift-technique-image-matching-python/

**Kernels**

In machine learning, a “kernel” is usually used to refer to the kernel trick, a method of using a linear classifier to solve a non-linear problem. It entails transforming linearly inseparable data like (Fig. 3) to linearly separable ones (Fig. 2). The kernel function is what is applied on each data instance to map the original non-linear observations into a higher-dimensional space in which they become separable.

**Batch Normalization**

Batch normalization is a technique for training very deep neural networks that standardizes the inputs to a layer for each mini-batch. This has the effect of stabilizing the learning process and dramatically reducing the number of training epochs required to train deep network
Batch normalization can be implemented during training by calculating the mean and standard deviation of each input variable to a layer per mini-batch and using these statistics to perform the standardization

Benefits:-
- Make neural networks more stable by protecting against outlier weights.
- Enable higher learning rates.
- Reduce overfitting.

**Stochastic Gradient**

A gradient descent algorithm in which the batch size is one. In other words, SGD relies on a single example chosen uniformly at random from a dataset to calculate an estimate of the gradient at each step.

**Mini-Batch SGD**

Mini-batch stochastic gradient descent (mini-batch SGD) is a compromise between full-batch iteration and SGD. A mini-batch is typically between 10 and 1,000 examples, chosen at random. Mini-batch SGD reduces the amount of noise in SGD but is still more efficient than full-batch.

**What is a T-Test?**

The t-test compares the means (averages) of two populations to determine how different they are from each other. The test generates a T-score and P-value, which quantify exactly how different each population is and the likelihood that this difference can be explained by chance or sampling error.
The bigger the t score, the larger the difference between samples, which also means the test results are more likely reproducible. 

**EXPECTATION MAXIMIZATION /GMM**

Maximum likelihood estimation is challenging on data in the presence of latent variables.
Expectation maximization provides an iterative solution to maximum likelihood estimation with latent variables.
Gaussian mixture models are an approach to density estimation where the parameters of the distributions are fit using the expectation-maximization algorithm.

E-Step. Estimate the missing variables in the dataset.
M-Step. Maximize the parameters of the model in the presence of the data

**t-Distributed Stochastic Neighbor Embedding (t-SNE)**

t-Distributed Stochastic Neighbor Embedding (t-SNE) is a non-linear technique for dimensionality reduction that is particularly well suited for the visualization of high-dimensional datasets

The algorithms starts by calculating the probability of similarity of points in high-dimensional space and calculating the probability of similarity of points in the corresponding low-dimensional space. The similarity of points is calculated as the conditional probability that a point A would choose point B as its neighbor if neighbors were picked in proportion to their probability density under a Gaussian (normal distribution) centered at A.
It then tries to minimize the difference between these conditional probabilities (or similarities) in higher-dimensional and lower-dimensional space for a perfect representation of data points in lower-dimensional space.
To measure the minimization of the sum of difference of conditional probability t-SNE minimizes the sum of Kullback-Leibler divergence of overall data points using a gradient descent method.

t-SNE and PCA
1. t-SNE takes longer to compute
2. PCA is a mathematical technique while t-SNE is probablistic
3. PCA can be fed into t-SNE to improve processing time
4. PCA is linear algorithm n t-SNE is non-linear

**Sampled Softmax**

When the dictionary is large,softmax computation for the whole dictionary might be time consuming. Therefore, the softmax is calculated only for certain words

**Difficult to train RNN bcs vanishing gradient and exploding gradient problem.**

Exploding gradient- clipping of the values 
Vanishing gradient- LSTM

**Resnet- Residual Neural Network**

Why do you need? Vanishing Gradient
Identity Mapping = What this means is that the input to some layer is passed directly or as a shortcut to some other layer. 
Only one fully connected layer at the end for output.

**GoogLenet**

Inception Module- Naive inception
Parallel filter operation on input received from layer
Split-Merge-Transform the data

Inception Module with a bottleneck -1x1 convolutions Bottleneck layer before passing it to expensive convolutions like 3x3 and 5x5
Reduces number of parameters and computational effort

Only one fully connected layer at the end

**Densenet**
Densely Connected networks or DenseNet is the type of network that involves the use of skip connections in a different manner. Here all the layers are connected to each other using identity mapping

**Reduce the size of feature maps**
 - strides 
 - avg pooling 
 - max pooling

**calculate output size of cnn**

you can use this formula [(W−K+2P)/S]+1.

W is the input volume 
K is the Kernel size 
P is the padding - 0 by default
S is the stride - default 1

**Advantages of using Convolution:**

1. Preserves and encodes spatial information 
2. Translation invariance
3. reduce in number of parameters compared to FC

**Advantages of pooling**
1. reduces computation 
2. maximum activation is used so you don't loose information
3. Translational invariance

**What's the difference between boosting and bagging?**

Boosting and bagging are similar, in that they are both ensembling techniques, where a number of weak learners (classifiers/regressors that are barely better than guessing) combine (through averaging or max vote) to create a strong learner that can make accurate predictions. Bagging means that you take bootstrap samples (with replacement) of your data set and each sample trains a (potentially) weak learner. Boosting, on the other hand, uses all data to train each learner, but instances that were misclassified by the previous learners are given more weight so that subsequent learners give more focus to them during training
Random Forest - Bagging - reduces variance thereby overfitting
Adaptive Boosting or Adaboost -Boosting - reduces bias, may tend to overfit

**Tensorflow**

Eager Execution:
TensorFlow's eager execution is an imperative programming environment that evaluates operations immediately, without building graphs: operations return concrete values instead of constructing a computational graph to run later.

In Tensorflow 2.0, eager execution is enabled by default.
Since there isn't a computational graph to build and run later in a session, it's easy to inspect results using print() or a debugger. 

```
def run(mean, std, sample_size, ex):
    if ex == EAGER:
        tf.enable_eager_execution()
    elif ex == INTERACTIVE:
        global interactive_session
        if interactive_session is None:
            interactive_session = tf.InteractiveSession()
    elif ex == SESSION:
        global session
        if session is None:
            session = tf.Session()
    else:
        assert ex in [INTERACTIVE, EAGER, SESSION]
```
 
The sequential API allows you to create models layer-by-layer for most problems. It is limited in that it does not allow you to create models that share layers or have multiple inputs or outputs.

Variables
A TensorFlow variable is the best way to represent shared, persistent state manipulated by your program. Variables are manipulated via the tf.Variable class. Internally, a tf.Variable stores a persistent tensor. Specific operations allow you to read and modify the values of this tensor. These modifications are visible across multiple tf.Sessions, so multiple workers can see the same values for a tf.Variable. Variables must be initialized before using.

Placeholders
A placeholder is a node (same as a variable) whose value can be initialized in the future. These nodes basically output the value assigned to them during runtime. A placeholder node can be assigned using the tf.placeholder() class to which you can provide arguments such as type of the variable and/or its shape. Placeholders are extensively used for representing the training dataset in a machine learning model as the training dataset keeps changing.

In tensorflow you create graphs and pass values to that graph. Graph does all the hardwork and generate the output based on the configuration that you have made in the graph. Now When you pass values to the graph then first you need to create a tensorflow session.

tf.Session()
Once session is initialized then you are supposed to use that session because all the variables and settings are now part of the session.

So, there are two ways to pass external values to the graph so that graph accepts them. One is to call the .run() while you are using the session being executed. Other way which is basically a shortcut to this is to use .eval(). I said shortcut because the full form of .eval() is

tf.get_default_session().run(values)
At the place of values.eval() run tf.get_default_session().run(values). You must get the same behavior, here what eval is doing, is using the default session and then executing run().

tf.nn.* - low level open
tf.layers.* - high level wrapper for the ops.Has more parameters like kernel_initializers etc

**Keras**

Alternatively, the functional API allows you to create models that have a lot more flexibility as you can easily define models where layers connect to more than just the previous and next layers. In fact, you can connect layers to (literally) any other layer. As a result, creating complex networks such as siamese networks and residual networks become possible.

**Losses**
1. Cross Entropy- also known as log loss
 L= -(ylog p + (1-y)log(1-p))
 Log losses penalizes for confidently predicting wrong answers. As in, the answer is wrong even though its predicted with high probability.
 multiclass cross entropy = - sigma(y log p)
 y is binary indicator (0 or 1) if class label c is the correct classification for observation 
 
2. hinge loss- SVM loss
L=max(0,1-y_pred*y)
The hinge loss function encourages examples to have the correct sign, assigning more error when there is a difference in the sign between the actual and predicted class values.

3. MSE- (y_pred-y)^2 . average this for whole dataset- The squaring means that larger mistakes result in more error than smaller mistakes, meaning that the model is punished for making larger mistakes.

4. MAE - average of |y_pred-y| - good with outliers

5. softmax loss - multiclass svm 
6. Binary loss

**The mathematics of matrix factorization**

Having discussed the intuition behind matrix factorization, we can now go on to work on the mathematics. Firstly, we have a set U of users, and a set D of items. Let \mathbf{R} of size |U| \times |D| be the matrix that contains all the ratings that the users have assigned to the items. Also, we assume that we would like to discover $K$ latent features. Our task, then, is to find two matrics matrices \mathbf{P} (a |U| \times K matrix) and \mathbf{Q} (a |D| \times K matrix) such that their product approximates \mathbf{R}:

R=PQ_Transpose

In this way, each row of {P} would represent the strength of the associations between a user and the features. Similarly, each row of {Q} would represent the strength of the associations between an item and the features. To get the prediction of a rating of an item d_j by u_i, we can calculate the dot product of the two vectors corresponding to u_i and d_j:

\hat{r}_{ij} = p_i^T q_j = \sum_{k=1}^k{p_{ik}q_{kj}}

Now, we have to find a way to obtain \mathbf{P} and \mathbf{Q}. One way to approach this problem is the first intialize the two matrices with some values, calculate how 'different’ their product is to \mathbf{M}, and then try to minimize this difference iteratively. Such a method is called gradient descent, aiming at finding a local minimum of the difference.

The difference here, usually called the error between the estimated rating and the real rating, can be calculated by the following equation for each user-item pair:


e_{ij}^2 = (r_{ij} - \hat{r}_{ij})^2 = (r_{ij} - \sum_{k=1}^K{p_{ik}q_{kj}})^2

Here we consider the squared error because the estimated rating can be either higher or lower than the real rating.
Learning rate is set to small value.This is because if we make too large a step towards the minimum we may run into the risk of missing the minimum and end up oscillating around the minimum.

**Naive Bayes**

A Naive Bayes Classifier is a supervised machine-learning algorithm that uses the Bayes’ Theorem, which assumes that features are statistically independent. The theorem relies on the naive assumption that input variables are independent of each other, i.e. there is no way to know anything about other variables when given an additional variable. Regardless of this assumption, it has proven itself to be a classifier with good results.
Given a vector x of features, Naive Bayes calculates the probability that the vector belongs to each class.

P( Ck | x1, x2, … xn )

Thanks to Bayes’ Theorem, we know that it is equal to:

P( Ck | x) = (P(x | Ck) *P(Ck) ) / P(x)

We know P(Ck) because of the class distribution in our data.

P(x | Ck) is equivalent to its joint probability P(Ck ,x1 , x2, …, xn). By the chain rule in probabilities we can expand it to:

P(Ck ,x1 , x2, …, xn) = P(x1 | x2, …, xn, Ck) * P(x2, …, xn, Ck)

= P(x1 | x2, …, xn, Ck) * P(x2, | x3, …, xn, Ck) * P(x3, …, xn, Ck)

= ….

= P(x1 | x2, …, xn, Ck) * P(x2, | x3, …, xn, Ck) * P(xn-1, | xn, Ck)* P(xn | Ck)* P(Ck)

We now make a strong assumption on the conditional probabilities we just calculated. We assume that they are conditionally independent. In other words, knowing that { xi+1, …, xn } occurred doesn't affect the probability of xi occurring. Put more formally:

P(xi, | xi+1, …, xn, Ck) = P(xi, | Ck).

This means that: P(Ck ,x1 , x2, …, xn) = 

To calculate each of the conditional class probabilities—P(xi | Ck )—we use a likelihood function to model the probability distribution. One of the most common is the Gaussian or normal distribution.

To make a prediction, we choose the class that had the highest score.

**How to handle Outliers??**

1. Visualization - Use a scatter plot and look for points that are far away
2. Interquartile Range- The good data is estimated to lie 25% to 75% of the values. 
You calculate the lower bound and upper bound. The values that lie outside this range are outliers
Lower Bound =Q1 -1.5*(Q3-Q1)(difference between starting value in 25% and ending value in 75%)
Upper Bound = Q3 + 1.5*(Q3-Q1)
Example: An outlier is any point that is below the lower bound or above the upper bound. As an example, let’s assume you have a data set of adult heights in inches. Let’s say the 25th percentile is 65 inches and the 75th percentile is 75 inches. That means that 50% of adult heights are between 65 and 75 inches, with an IQR of 10 inch

Effect of Outliers:
The average of the data is moved because moved. The mean cannot be used to estimate the avg value as it could be biased by the outlier.
We need to use median in such cases to avoid the effect of outliers.

**Decision Trees**

You look for a split that classifies data into a class completely or atleast the best.
Gini Index= 1-summation(P(k=c))
Entropy= - p log p

Outliers should not be removed until it erroneous  values.

**Dice Coefficient**

2 * |X| ∩ |Y| / (|X| + |Y|)
|X| means the number of elements in set X

**Triplet Loss Function**

triplet loss function using which gradients are calculated. The variable “a” represents the anchor image, “p” represents a positive image and “n” represents a negative image. We know that the dissimilarity between a and p should be less than the dissimilarity between a and n,. Another variable called margin, which is a hyperparameter is added to the loss equation. Margin defines how far away the dissimilarities should be, i.e if margin = 0.2 and d(a,p) = 0.5 then d(a,n) should at least be equal to 0.7. Margin helps us distinguish the two images better.
L=max(0, d(a,n)-d(a,p)+margin)

**bias**

The bias value allows the activation function to be shifted to the left or right, to better fit the data.

**GRU**

A slightly more dramatic variation on the LSTM is the Gated Recurrent Unit, or GRU, introduced by Cho, et al. (2014). It combines the forget and input gates into a single “update gate.” It also merges the cell state and hidden state, and makes some other changes. The resulting model is simpler than standard LSTM models, and has been growing increasingly popular.

**Activation Functions**

Activation :
sigmoid- (0,1)
(1/e^-x+1)
Problem = Vanishing Gradient

Tanh=[-1,1]
advantage: Gradient is steeper compared to sigmoid and larger range of values
problem:problem of gradients at the ends of the function continues.

ReLU=  max(0,x)
The fact that the calculation load is less than the sigmoid and hyperbolic tangent functions has led to a higher preference for multi-layer networks
Vanishing gradient
Cheap to compute as there is no complicated math and hence easier to optimize
It converges faster. It accelerates the convergence of SGD compared to sigmoid and tanh (around 6 times).
Not have vanishing gradient problems like tanh or Sigmoid function
It is capable of outputting a true zero value allowing the activation of hidden layers in neural networks to contain one or more true zero values called Representational Sparsity

Problems with ReLU
The downside for being zero for all negative values called dying ReLU. So if once neuron gets negative it is unlikely for it to recover. This is called “dying ReLU” problem
If the learning rate is too high the weights may change to a value that causes the neuron to not get updated at any data point again.
ReLU generally not used in RNN because they can have very large outputs so they might be expected to be far more likely to explode than units that have bounded values.

Leaky ReLU- max(0.01x,x)

softmax- Multiclass classification
=e^x/summation(e^x)
The sigmoid function can be used if you say that the hyperbolic tangent or model can be learned a little slower because of its wide range of activating functions. But if your network is too deep and the computational load is a major problem, ReLU can be preferred. You can decide to use Leaky ReLU as a solution to the problem of vanishing gradients in ReLU. But you do more computation than ReLU.

**Problem with using RNN for translation**

1. Memory Problems - difficult to remember the context for long sentences. Solved using LSTM 
2. Tough to converge- Tough to converge even with LSTM
3. Slow training time- has to be trained sequentially

**Neural Machine Translation(NMT):**

1. Encoder - Decoder architecture
2. each word in base language is passed to the RNN. the result of this and new word is fed to next RNN in encoder
3. The output vector from the encoder is then provided to decoder as input

**Attention is all you need: Transformer Network**

Attention - focus on things with high resolution.
No sequential proccessing. So model converges faster.

Attention is based on query,key and value.
Attention weights for a word is calculated as Q.K_T . This value is scaled and softmaxed and then matmul with VT

**NLP vs NLU vs NLG**
<img width="473" alt="image" src="https://github.com/user-attachments/assets/df255a1b-aaa5-4fa4-9c98-4a1bc17da905">

**RAG- Retrieval Augmented Generation**
RAG operates by fetching relevant documents or data snippets based on a query and then using this retrieved information to generate a coherent and contextually appropriate response. This method is particularly valuable in fields like chatbot development, where the ability to provide precise answers derived from extensive databases of knowledge is crucial.

```
from transformers import RagTokenizer, RagSequenceForGeneration
# Replace with your chosen model and tokenizer names
model_name = "facebook/rag-token-base"
tokenizer = RagTokenizer.from_pretrained(model_name)
model = RagSequenceForGeneration.from_pretrained(model_name)
inputs = tokenizer(your_prompt, return_tensors="pt")
retrieval_output = model.get_retrieval_vector(inputs)
generation_inputs = {
    "input_ids": inputs.input_ids,
    "attention_mask": inputs.attention_mask,
    "retrieval_logits": retrieval_output,
}
generation_output = model.generate(**generation_inputs)
generated_text = tokenizer.decode(generation_output.sequences[0])
```


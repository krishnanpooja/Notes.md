**How do we form a confidence interval?**	

The purpose of taking a random sample from a lot or population and computing a statistic, such as the mean from the data, is to approximate the mean of the population. How well the sample statistic estimates the underlying population value is always an issue. A confidence interval addresses this issue because it provides a range of values which is likely to contain the population parameter of interest.

**Confidence Level**

Confidence intervals are constructed at a confidence level, such as 95 %, selected by the user. What does this mean? It means that if the same population is sampled on numerous occasions and interval estimates are made on each occasion, the resulting intervals would bracket the true population parameter in approximately 95 % of the cases. A confidence stated at a 1−α level can be thought of as the inverse of a significance level, α.

**Oversampling and Undersampling**

The two main approaches to randomly resampling an imbalanced dataset are to delete examples from the majority class, called undersampling, and to duplicate examples from the minority class, called oversampling.

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

Underfitting occurs when a statistical model or machine learning algorithm cannot capture the underlying trend of the data.  Intuitively, underfitting occurs when the model or the algorithm does not fit the data well enough.  Specifically, underfitting occurs if the model or algorithm shows low variance but high bias.  Underfitting is often a result of an excessively simple model.

**Difference between SVM and Logistic Regression**

SVM tries to finds the “best” margin (distance between the line and the support vectors) that separates the classes and this reduces the risk of error on the data, while logistic regression does not, instead it can have different decision boundaries with different weights that are near the optimal point.
SVM works well with unstructured and semi-structured data like text and images while logistic regression works with already identified independent variables.
SVM is based on geometrical properties of the data while logistic regression is based on statistical approaches.
The risk of overfitting is less in SVM, while Logistic regression is vulnerable to overfitting.
SVM works better with higher dimensional data unlike logistic regression which is good with low dimensional.
SVM uses few data points called support vectors unlike logistic regression which uses the entire training data.

Problems that can be solved using SVM
Image classification
Recognizing handwriting
Cancer detection

Problems to apply logistic regression algorithm
Cancer detection — can be used to detect if a patient have cancer (1) or not(0).
Test score — predict if a student passed(1) or failed(0) a test.

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
4.  repeat until convergence

**Generative and discriminative model**

Generative - models actual distribution of each class. P(y,x) - joint probability-- [learn stuff indepth]
‌Naïve Bayes
Bayesian networks
Markov random fields
‌Hidden Markov Models (HMM)

Discriminative - models the decision boundary between the classes. P(y|x) -- conditional probability--[look for differences]
‌Logistic regression
Scalar Vector Machine
‌Traditional neural networks
‌Nearest neighbour
Conditional Random Fields (CRF)s
If data is highly correlated, we should expect many σᵢ values to be small and can be ignored.





## Introduction to Tensorflow model analysis (TFMA)
Its is an open source scalable framewok for doing deep analysis of model performance including analysing on small data. 
- Ensures that model meet required quality thresholds

#### Architecture
- datatype is tfma.extracts is a dictionary

<img width="838" alt="image" src="https://github.com/krishnanpooja/Notes.md/assets/8016149/c43cc005-0b20-426e-8a7d-434dee2ae08f">

- Tensorboard is used to analyase the training process while TFMA is used for trained model analysis
- Tensorboard retrieves metrics for mini batch streaming data while TFMA uses Apache beam for scaling on the datatsets and gives evaluation results after running on the entire dataset

  #### Steps involved in TFMA code
  1. Export EvalSavedModel for TFMA
     <img width="793" alt="image" src="https://github.com/krishnanpooja/Notes.md/assets/8016149/3e1a79db-0550-43f5-9064-80ad18021459">

  2. Create EvalConfig
  <img width="883" alt="image" src="https://github.com/krishnanpooja/Notes.md/assets/8016149/e6c171df-c5f4-4063-b7da-df55d13077b9">

  3. Analyze the model
 <img width="831" alt="image" src="https://github.com/krishnanpooja/Notes.md/assets/8016149/075adeb5-7b17-48a8-9610-8ed48ea71ec4">

  4. Visualize the model
 <img width="506" alt="image" src="https://github.com/krishnanpooja/Notes.md/assets/8016149/1acb42f3-d866-4b3f-8aa7-3c8590913f3d">

  
### Continuous monitoring 
Types of dataset shift:
1. covariate shift - distribution of the input changes but the label remains the same
2. prior prababilty shift - label distribution changes -concept drift

### Methods to control and monitor models
#### Supervised method
1. __Statistical process control__ -   It assumes that the stream of data will be stationary, which it may or may not be depending on your application. That the errors follow a binomial distribution. It analyzes the rate of errors and since it's a supervised method, it requires us to have labels for incoming stream of data. Essentially, this method triggers a drift alert if the parameters of the distribution go beyond a certain threshold rule.

2. __Sequential Method__ -
 - In sequential analysis, we use a method called linear four rates. The basic idea is that if data is stationary, the contingency table should remain constant.
 - The contingency table, in this case, corresponds to the truth table for a classifier that you're probably familiar with: true positive, false positive, false negative, and true negative. You use those to calculate the four rates: net predictive value, precision, recall, and specificity. If the model is predicting correctly, these four value should continue to remain constant.

3. __Error Distribution Monitoring__
   - The method of choice here is known as adaptive windowing.
   - . In this method, you divide the incoming data into windows, the size of which adapts to the data. Then you calculate the mean error rate at every window of data. Finally, let's calculate the absolute difference of the mean error rate at every successive window and compare it with a threshold based on the Hoeffding bound. The Hoeffding bound is used for testing the difference between the means of two populations.

#### Unsupervised 
1.__Clustering/Novelty detection__
   - In this method, you cluster the incoming data into one of the known classes. If you see that the features of the new data are far away from the features of known classes, then you know that you're seeing an emerging concept
   - multiple algorithms - OLINDDA, MINAS, ECSMiner, and GC3.
   - While the visualization and ease of working with clustering work well with low dimensional data, the curse of dimensionality kicks in once the number of dimensions grow significantly. Eventually, these methods start getting inefficient, but you can use dimensionality reduction techniques like PCA to help make it manageable

2. __Feature distribution monitoring__
 - we monitor each __feature__ of the dataset separately.
 - You split the incoming dataset into uniformly size windows, and then compare the individual features against each window of data.
 -  The first is Pearson correlation, which is using the change of concept technique. There's also the Hellinger distance, which is used in the Hellinger Distance Drift Detection Method or HDDDM. The Hellinger distance is used to quantify the similarity between two probability distributions. See the reference list at the end of the week for more information.
 -   The first is Pearson correlation, which is using the change of concept technique. There's also the Hellinger distance, which is used in the Hellinger Distance Drift Detection Method or HDDDM. The Hellinger distance is used to quantify the similarity between two probability distributions. See the reference list at the end of the week for more information.

3. __Model-dependent monitoring__
- This method monitors the space near the decision boundaries or margins in the latent feature space of your model.
- One of the algorithms used is Margin Density Drift Detection or MD3.
- Space near the margins where the model has low confidence matters more than in other places, and this method looks for incoming data that falls into the margins. A change in the number of samples in the margin, the margin density, indicates drift.
- This method is very good at reducing the false alarm rate.

### Contiuous Evalution on Cloud
 - One such service is Microsoft Azure Machine Learning datasets which focuses on data drift.
- Another is Amazon SageMaker Model Monitor which focuses on concept drift.
- Google Cloud AI continuous evaluation which also focuses primarily on concept drift,
Google Cloud AI continuous evaluation regularly samples prediction input and output from trained models that you've deployed to AI Platform Prediction. AI Platform Data Labeling service then assigns human reviewers to provide ground truth labels for a sample of the prediction requests that you receive. Or alternatively you can provide your own ground truth labels using a different technique.

 
 ### Model Remediation
 1. Data augmentation - adding synthetic data into training set and correct unbalanced training data
 2. Interpretable and explainable ML - understan neural networks instead of considering them as black box  and also understand how th edata gets transformed
 3. Model editing - applied to decision trees which involves manual tweaks to your use case
 4. Model assertions- implement business rules that override model predicitions like age cannot be a negative value

### How to avoid model bias
- __Discrimination Remediation__
  - include people from different background
  - conduct feature selection on training data
  - use fairness metrics to select hyperparams and decision cut-off thresholds

## Fairness
1. Positive and Negative rates - Prcentage of data points classified as positive and negative
2. True Positive rate / False Negative Rate
3. True negative rate/ False positive rate
  -- FPR- percentage of negative data points that are incorrectly labeled positive
  -- use case : where misclassifying something as positive is more concerning than classifying the positives
4 Accuracy and AUC
 - The percentage of data points that are correctly labeled when each class is given equal weight independent of number of samples
 - Use case: precision is critical.
- 
  


 6. 

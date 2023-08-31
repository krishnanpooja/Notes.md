### Learning Objectives
- Describe the differences between ML modeling and a production ML system
- Identify responsible data collection for building a fair production ML system
- Discuss data and concept change and how to address it by annotating new training data with direct labeling and/or human labeling
- Address training data issues by generating dataset statistics and creating, comparing and updating data schemas


## Collecting,labeling and validating data
ML development + Software development

Modern Software Development concepts applied in ML pipelines:
1. Scalabilty
2. Extensibilty - can you add new stuff?
3. Configuration
4. consistency and reproducibilty
5. safety and security
6. modularity
7. testabilty?
8. monitoring
9. best pratices

### ML Pipelines

<img width="836" alt="image" src="https://github.com/krishnanpooja/Notes.md/assets/8016149/f8aa613b-0042-40e2-8a44-078ee3cb4c2e">

ML pipelines are directed acyclic graphs (DAGS) (graphs without any circles)

<img width="952" alt="image" src="https://github.com/krishnanpooja/Notes.md/assets/8016149/41748799-4b9e-4628-b56e-5283876ba023">

<img width="788" alt="image" src="https://github.com/krishnanpooja/Notes.md/assets/8016149/ce7041a2-a816-4d0c-bb56-bc3a4b7d344a">

### Collect Data
Quality of model depends on the quality of data

1. Understand the user. The user needs is converted into data needs, feature needs, label needs
2. Get to know your data.
     - Identify data sources
     - check if they are refreshed
     - consistency for values, units and data types
     - monitor outliers ad errors

#### Dataset issues
1. Inconsistent formatting -like zero , 0 , 0.0
2. Compounding errors from other ML models
3. Monitor data sources for system issues and outages

Feature Engineering helps maximiize the predictive signals
Feature selection helps to measure the predictive signals

#### Degraded Model Performance
1. Slow- concept or data drift - over time things change and this could affect the perforamce of the system
2. Fast - Bad sensor, bad los and bad softare update

#### Data and concpet change
###### Easy problems:
- Ground Truth changes slowly(in matter of months/years)
- Model retraining driven by:
       - Model improvements, better data
       - changes in softare and /or systems
###### Harder Problems
- Ground truth changes faster (in matter of weeks)
- Model retraining driven by:
       - Declining model performance
       - Model improvements, better data
       - changes in softare and /or systems
   
###### Really Hard Problems
- Ground truth changes really fast (in matter of days,hours,min)
- Model retraining driven by:
       - Declining model performance
       - Model improvements, better data
       - changes in softare and /or systems
   
#### Process Feedback and Human Labeling
Data Labeling:
1. Process Feedback - Example: actual vs predicted click-through - label positive if predicted click-through=actual 
2. Human Labeling -ex:- cardiologists labeling MRI images

Process feedback- Open-Source log analysis tools
1. Logstash - Ingests data from multitude of sources , transfroms it and sends to your fav "stash"
2. Fluentd- Open source, unify data collection and consumption
On cloud:
1. BindPlane -google
2. AWS ElasticSearch
3. Azure Monitor

Human Labeling:
Peopele ("raters") to examine data and assign labels manually
Slow , difficult and expensive process

#### Detecting Data Issues
##### Data Issues:
__Drift__ - Changes in data over time such as data collected once a day
__Skew__ - Difference between two static versions, or dif sources such training set and serving set

Detecting data issues:
1. Detecting schema skew - training and serving data do not conform to the same schema (like you may train on int but later get string or float)
2. Detetcing Distribution skew- Dataset shift-> vovariate or concept shift
3. Requires continuous evaluation

<img width="482" alt="image" src="https://github.com/krishnanpooja/Notes.md/assets/8016149/4b9951e6-0240-4055-b99a-a7ff51ca2ede">


#### TensorFlow Data Validation(TFDV)
<img width="869" alt="image" src="https://github.com/krishnanpooja/Notes.md/assets/8016149/2c5161e7-fb37-4c4b-8769-4b07fc71030b">
  
##### Schema Skew
Serving and training data don't conform to same schema. int!=float

##### Feature Skew
Training feature values are different from serving feature values
1. features values are modified btw training and skew time
2. transformation applied only in one of the two instances

#### Distribution skew
Distribution btw the two is significantly different:
1.fault sampling method during training
2. diff data sources for training and serving data
3. Trend, seasonality changes in data over time



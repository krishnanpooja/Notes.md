## Introduction to Preprocessing
Squeeze most out of the data:
1. make data useful before training
2. normalize data to help models learn
3. incrase predictive quality - transforming data 
4. reduce dimensonality with feature engineering

#### Feature Engineering
Feature engineering tries to improve model's abilty to learn and reduces the cost invilved in the process by merging / removing featrues.
During training, since whole data is present we can normalize the data. This should be applied while serving as well.

### Preprocessing Operations
Main preprocessing operations:
1. Data Cleansing
2. Feature Tuning
3. Representation Transformation
4. Feature Extraction
5. Feature Construction


### Feature Engineering Techniques
<img width="527" alt="image" src="https://github.com/krishnanpooja/Notes.md/assets/8016149/49309731-2895-4c96-87a7-56a00fd80787">

_Scaling_
Convert values from their natural range into a prescribed range
ex:- gray scale from [0,255] is rescaled to [-1,1]
_Benefits_
1. Helps neural nets converge fatser
2. do away with NaN errors during training
3. for each feature, model learns the right weights

_Normalization_

<img width="344" alt="image" src="https://github.com/krishnanpooja/Notes.md/assets/8016149/939706ea-a4fd-4395-a5cf-f4075b248aaa">

_Standardization_ (Z score)

<img width="535" alt="image" src="https://github.com/krishnanpooja/Notes.md/assets/8016149/45a6090d-c2b9-4550-82a9-6477254bfc0f">

_Bucketizing/Bining_

<img width="663" alt="image" src="https://github.com/krishnanpooja/Notes.md/assets/8016149/14104e16-be85-498a-bc57-5c8c7f2a8389">

_other Techiniques_
1. PCA
2. t-SNE

#### Feature Transformation at Scale
### Feature Crossing
Combine multiple features together into a new feature
- Encodes nonlinearlity in the feature space or encodes the same information in fewer feaatures
- [AxB]: multiplying the values of two features

## Preprocessing at Scale
### Tensorflow Transform
Statistic Gen- computes statistics like mean etc.. 
Transform compnent does the featre engineering. 

<img width="929" alt="image" src="https://github.com/krishnanpooja/Notes.md/assets/8016149/df2fcd7e-15db-4e86-881c-043753e03fda">
The transform graph contains the transform operations performed on the data and transform data is the output.

<img width="920" alt="image" src="https://github.com/krishnanpooja/Notes.md/assets/8016149/f7579fcc-c013-4b2c-a8b8-c15a65a88101">

How Transfrom applies feature transfromations
Graph in left for feature Eng, analysis across dataset to collect constants to apply to transform graph so as to transform induvidual examples on training data and the same can be applied on serving data. So no skew!
<img width="910" alt="image" src="https://github.com/krishnanpooja/Notes.md/assets/8016149/fb0f9d19-85fe-493f-b2af-c38d6e57ed22">

 <img width="933" alt="image" src="https://github.com/krishnanpooja/Notes.md/assets/8016149/d07edddf-815a-4f55-9851-67f56b10f9f6">
 <img width="573" alt="image" src="https://github.com/krishnanpooja/Notes.md/assets/8016149/32ce3553-f190-4d4a-b0b5-cedb625b5921">

## Steps involved in transforming raw data
<img width="865" alt="image" src="https://github.com/krishnanpooja/Notes.md/assets/8016149/080ae4e3-32c9-4968-bb6c-6c8fb903fe9b">
<img width="878" alt="image" src="https://github.com/krishnanpooja/Notes.md/assets/8016149/cb200002-0e92-4f1b-929f-4f3afd3fe742">

```
# Ignore the warnings
tf.get_logger().setLevel('ERROR')

a temporary directory is needed when analyzing the data
with tft_beam.Context(temp_dir=tempfile.mkdtemp()):
    
    # define the pipeline using Apache Beam syntax
    transformed_dataset, transform_fn = (
        
        # analyze and transform the dataset using the preprocessing function
        (raw_data, raw_data_metadata) | tft_beam.AnalyzeAndTransformDataset(
            preprocessing_fn)
    )

unpack the transformed dataset
transformed_data, transformed_metadata = transformed_dataset

print the results
print('\nRaw data:\n{}\n'.format(pprint.pformat(raw_data)))
print('Transformed data:\n{}'.format(pprint.pformat(transformed_data)))
```

ingest data from a base directory with __ExampleGen__
compute the statistics of the training data with __StatisticsGen__
infer a schema with __SchemaGen__
detect anomalies in the evaluation data with __ExampleValidator__
preprocess the data into features suitable for model training with __Transform__

If several steps mentioned above sound familiar, it's because the TFX components that deal with data validation and analysis (i.e. StatisticsGen, SchemaGen, ExampleValidator) uses Tensorflow Data Validation (TFDV) under the hood.

#### Create the Interactive Context
When pushing to production, you want to automate the pipeline execution using orchestrators such as Apache Beam and Kubeflow. You will not be doing that just yet and will instead execute the pipeline from this notebook. When experimenting in a notebook environment, you will be manually executing the pipeline components (i.e. you are the orchestrator). For that, TFX provides the Interactive Context so you can step through each component and inspect its outputs.


## Feature Selection
#### Feature Space
N dimensional space is defined by N features
Not including the target label

Ensure the feature space coverage :
- Data affected by: seasonality, trend, data drift
- serving data: new values in features and labels (concept drift)
In these cases we need to reiterate the ML pipeline as the model needs to learn the new version

#### Feature Selection
- Identify the features that best represent the relationship
- therby reducing the size of feature space (lentioned above0
- thereby reducing the resource requirements and model complexity

__Advantages__:
1. reduce the storage cost
2. reduce i/o requirements
3. minimize training and inference costs

<img width="422" alt="image" src="https://github.com/krishnanpooja/Notes.md/assets/8016149/b2002fcb-6bc6-4d24-a9a2-0a03813cb4dc">

   

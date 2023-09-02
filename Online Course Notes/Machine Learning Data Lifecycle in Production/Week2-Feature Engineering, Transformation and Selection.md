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


  

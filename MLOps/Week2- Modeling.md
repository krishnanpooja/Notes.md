## Modeling
Problems faced:
-> navigational queries should return the correct website else the search engine looses its value
-> rare classes  - skewed data distribution(99% negative and 1% positive)

#### Establish a baseline
- human level performance as the baseline while evaluting the model. This may help focus on topics that actually require attention.
- unstructured data  - image, audio, nlp - measure human level performace as baseline here
- structured data - ecomm data (user, purchase...) - data stored in database 

#### ways to establish baseline
- human level performance
- literature search
- quick and dirty implementation
- performance of old systems


 - try to overfit on the small training system
 
 #### Error analysis and performance auditing
 ##### Error analysis example
 - look through the wrongly predicted data and find params that could have gone wrong for that prediction- like low bandwidth, car noise, people noise
 etc.
 - iteratively brain storming on the error analysis
 
 
 ### Performace Auditing 
 Check for fairness/bias, accuracy and other possible problems
 - mean accuracy for different genders and major accents
 - check prevalance of offensive words in output
 - 
 
 
 

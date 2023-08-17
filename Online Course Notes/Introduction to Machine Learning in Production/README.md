## Week1-Introduction
### Learning Objectives
- Identify the key components of the ML Lifecycle.
- Define “concept drift” as it relates to ML projects.
- Differentiate between shadow, canary, and blue-green deployment scenarios in the context of varying degrees of automation.
- Compare and contrast the ML modeling iterative cycle with the cycle for deployment of ML products.
- List the typical metrics you might track to monitor concept drift.


## The Machine Learning Project Lifecycle
Two major problem:
1. concept drift or data drift - where a new example is seen that model was not trained on
2. ML code is way smaller than actual deployment production code as it contains info like feature extraction, data verification, analysis tool etc.

#### Steps of ML Project
<img width="935" alt="image" src="https://github.com/krishnanpooja/Notes.md/assets/8016149/0181a00a-c0c7-4307-bfd5-2e6b192d718a">
            
1. Scoping - Define project
 - Identify Key metrics
 - like for speech recognition- accuracy, latency and throughput
   
2. Data- Define data and establish baseline
        - Label and organize data
- data labeled consistently?
- how much silence before /after each clip?
- how to perform volume normalization?

3. Modeling- Select and train model
           - Perform error analysis

- ML system=code+hyperparamters+data
4. Deployment - Deploy in production
            - Monitor and maintain the system


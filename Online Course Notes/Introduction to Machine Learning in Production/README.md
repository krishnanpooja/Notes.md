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

### Deployment
Major problems:
1. Concept Drift - if X->Y relation changes. Ex:- due to inflation the price of the house 'of same size' changes
2. Data drift - if X changes. i.e, if a person say an actor becomes very famous suddenly.
3. Software Engineering issues -
Points to consider when building software applications:
   1. Realtime or batch decision-like fast predicition in 100ms or overnight computation
   2. Run on Edge/web browser or cloud -(Edge-loal system that can be used without wifi)
   3. Compute resources (CPU/GPU/memory)
   4. latency, throughtput (queries per second)
   5. logging - log data for analysis
   6. security and privacy
      
### Deployment Patterns
1. Shadow mode deployment- where the learning algo shadow's the human. This helps in collecting data and verify the performance of the algo before deploying it
2. Canary deployment-Roll out small traffic initially and ramp up traffic gradually
3. Blue green deployment - Old or blue version is the older version of the software. In this pattern, there exists a 'router' that routes the images/data samples to  blue version of software and then suddenly shifts to the new version. ADVANTAGE:- Easy rollback in case of errors.

### Monitoring
how to monitor a machine learning system?
1. Dashboard- like
             - Software metrics - log server load, franction of non-null outputs
             - input metrics like Avg input length, avg input volume
             - output metrics - # times return "", # times the suer redoes search
   - threshold need to be set to trogger alarm

### Pipeline Monitoring
VAD- Voice Activity Detection - module recognises if someone is talking - it reduces the bandwidth sent to the cloud server

### Metrics to Monitor
#### Monitor
1. Software metrics
2. Input metrics
3. output metrics

#### How quickly does the data change?
User data generally changes slowy. Exception example covid-19 caused change in people's shopping behaviour
Enterprise data can change quickly - like factory data etc.


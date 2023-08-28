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





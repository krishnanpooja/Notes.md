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

  

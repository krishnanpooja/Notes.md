**WEEK 1**

## Introduction to Machine Learning
An edge device is a device that provides an entry point into enterprise or service provider core networks. Examples include routers, routing switches, integrated access devices, multiplexers, and a variety of metropolitan area network and wide area network access devices.
Example: camera as part of edge device network, receives the photo of a phone (detect whether defective or not). sends the image to _prediction server_ (accepts it in form of API calls) which then sends to the control software

#### Steps of an  ML Project
- scoping - define the project- x and y in project
- data - acquire data - establish baseline, origanise the data, 
- modeling- train the model - train, error analysis, collect more data
- deployment - deploy and monitor & maintain the system

#### Case Study: Speech Recognition
###### Scoping
- Voice Search
- Key Metrics:
--- Accuracy, latency, throughput

###### Data
- Data label consistency
---- Which of this sample is the correct transcription - um... todaty's weather, um today's weather, today's weather
- how much <sil> before/after?
- volume normalization- high speakers, low speakers,...

###### Modeling
  - model code
  - hyperparameters
  - data

  **Phone**-> uses micrphone-> uses VAD module (Voice Activity Detection)------> _Speech API_-----> **Prediction Server** ------> _Transcript & result _---> **Phone** (Front end code)
  

  
 ## Deployment 
### Concept or data drift
  
New analomous data at the production end. Example: model trained on adult voices subjected to teenagers voice in production env degrades the system performance.
  concept drift = x->y mapping changes
  data drift = x changes

 ### software engineering issues
  - realtime or batch - realtime prediction or batch predictions required?
  - cloud vs edge/browser -edge devices are useful when internet is compromised
  - compute resources (GPU/CPU/memory)
  - Latency or throughput 
  - logging (log as much as possible for analysis)
  - security and privacy

 ## Deployment Patterns
 
 - Gradual ramp up with monitoring 
 - Rollback
  
**shadow mode**: human inspector inspects the machine learnign models predictions
  
**Canary deployment**: provide only minimum (5%) of the data traffic to the model to predict on. helps recognise the problems in learning algorithm before hand.
  
**Blue green deployment**: the data traffic is sent to different versions of software, by the router. The old version called blue and the new version.  is called green. Easy to rollback.
 
**Degree of automation**: human only-> shadow mode-> AI assitance (like drawing bounding boxes)-> partial automation -> full automation
  
  
  
  

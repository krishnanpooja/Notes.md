# Table of Contents
1. [Provide clear instructions](#Provideclearinstructions) 
1. [Give the model time to think](#Givethemodeltimetothink)  
  
 **Principles**
 - Provide clear instructions- longer prompt could be more informative, 
 - Give model time to think
 
 **OpenAI - Python Library - !pip install openai**
 
 ```
 import openai
 openai.api_key = "sk-" # set it as env variable
 
 ```
 <a name="Provideclearinstructions"></a>
 **Provide clear instructions**
- longer prompt could be more informative, Tatics:

***Tatic 1. delimiters*** 

    1.Triple quotes: """,

    2.Triple backticks: ''',

    3.Triple dashes: ---,

    4.Angle brackets: <>,

    5.XML Tags: <tag> </tag>,

***Tatic 2. Ask for structured output***
HTML, JSON

***Tatic 3: condition based***
***Tatic 4: Few shot prompting****
You give the model successful examples of completing the tasks. Then ask the model to perform the task.
  
 <a name="Givethemodeltimetothink"></a>
 **Give the model time to think**



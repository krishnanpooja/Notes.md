# Effective Prompt Design and Useful framework for building LLM Application

Difference of ChatGPT from previous GPT version - *how it takes input*

## Prompt Tip
1. Explain the task in simple terms. Don't use jargons as models are trained on web data
2. Breaking the problem down
3. Make the model repeat the model input.
4. Recency Bias- It gives more importance to instruction given at the end.
- you can avoid this by repeating the most important instruction(s) at the end of the prompt
- like for few shot, the classification task can be biased in the order of the labels. You can reduce importance of a class which is oversampled by mentioning it in the beginning
- 5. Chain of thought prompting - Instruct the model to proceed step-by-step and present all the steps involved

## Model Parameters - Based on OpenAI Studio
  Adjust these parameters to improve model behaavior
  1. Max response - Doesn't affect quality of the repsonse. Just affects the cost and latency
  2. Temperature - Most important parameter.
     Temperature=0=precise
     Temprature>0.5 = creative
  3. Toy P-How many tokens to take
  4.  Frequency penalty- if you don't want it to repeat set it to 0
  5.  Presence penalty - if you don't want it to genearte same words as in prompt set it to 0.

## Prompt Chaining 
Entity Extraction -> entity are extracted 
Summarization-> prompt sumaarizes the the output from entity
Sentiment Analysis-> output from summary is used to figure out sentiment

## Meta Prompt
Advanced way of writing prompts
1. The conversational agent whose code name is Dana -> what is expected of the agent, features like what she understands
2. Capabilties of the agent -> like repsonses should be informational and logical
3. where it can gather information from -> whether it can access DBs, API etc
4. Safety matters for responsible AI

+

Prompt 
ex:- write tagline for ice cream shop

Result: Scoops of heaven in heart of Phenoix


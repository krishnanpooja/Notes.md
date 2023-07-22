# LLM- Large Language Models

How do they work?
- Generative pre-trained transformer (GPT)
  - "pre-trained' so it means that the model has been trained up until a point in time and is not continuously training.
  - it does not have access to Internet
  - GPT generates one word at a time based on probabilties of those words showing up next to each other
  - They are probabilistic model where in each word is not known until its produced making it risky and irrepeatable and can give different output for the same question
  - Capabilties- Generative, Transformation (text format to image), summarization, Question answering, conversation, classification-sorting, categorizing
  - GPT is cross functional- use one of capabilties or combine them and use

  ## Prompt Basics
  - Basic Prompt *(zero shot)*- like asking a question to the model.More descriptive, better output
  - *Few shot*-Example input and output. Showing the model what the desired output looks like
  - 
  ## Controls
  - Temperature- lower- determinisitic and higher value-more probablistic
  - Tokens and memory - Model breaks input and output into chunks called tokens. There is a limit to number of tokens the model can use at any time. Current solution for this include askign the model to make summaries of the chunks of the novel but this may lead to data loss.

  ## Pretrained vs Grounding data
    - like asking for number of vacations days in general vs providing a profile of the person in Microsoft and asking for number of vacation days.

  ## UX Design
  - A collaborative UX puts an emphasis on augmentation over automation.
  - The collborative UX requires us to always position the user in the driving seat so they can confidently and iteratively guide the model
  - 


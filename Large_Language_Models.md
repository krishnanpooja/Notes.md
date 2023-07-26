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
  - Fine Tuning - The model is trained via repeated gradient updates using a large corpus of example tasks
  - 
  ## Controls
  - Temperature- lower- determinisitic and higher value-more probablistic
  - Tokens and memory - Model breaks input and output into chunks called tokens. There is a limit to number of tokens the model can use at any time. Current solution for this include askign the model to make summaries of the chunks of the novel but this may lead to data loss.

  ## Pretrained vs Grounding data
    - like asking for number of vacations days in general vs providing a profile of the person in Microsoft and asking for number of vacation days.

  ## UX Design
  Continuous input and output loop
  - A collaborative UX puts an emphasis on augmentation over automation.
  - The collborative UX requires us to always position the user in the driving seat so they can confidently and iteratively guide the model
  - Customized input- Like in Outlook we might allow users to set the tone of their response with some pre-defined options like "serious"
  - Show how how the inputs and outputs interact with each other
  - keep history of inputs and outputs - like new output may make the results worse so user can use history to improve
  - increase Friction- if a work can be done in 2 steps still force the users to cross check and slow users dowm as LLMs are probablistic
  - Aloowing user to step in and edit from the output is a crucial ascpect of collaborative UX. It makes the model a Aumenter, helping the user and at the same time keeping them in driving seat
  - LLMs can output wrong/inaccurate information which can be difficult to spot. Consider ways to flag potiential mismatches or issues between the source and LLM output.
  - Use citations- can control hallucinations to some extent that it gets from pre trained data
  - Encourage fact checking-power users to fact check without spending tons of time on it.
  - Emphasize LLMs role and limitations -educate users on limitations like warn them about the risks of invesitng in stock market etc
  - Feedback- Report inappropriate outputs- 

Other key moments:- Other than input and output
- Pre-experience setting expectations- Make use of landing pages, pre-release narrativesand sig up mechanisms to help set the right expectations.
 First experience- build appropriate trust

## Content Guidance
- Avoid Humanizing AI
- When and how to use "we"
  - We = AI+Customer
  - We!=Microsoft
- Useage of 'I'

HAX toolkit- Human AI Experience 
Guidelines:
1. Make clear what the system can do
2. Make clear how well the system can do what it can do
3. Make clear why the system did what it did

## LLM accessibilty
- accessible without any discrimination to all
- 

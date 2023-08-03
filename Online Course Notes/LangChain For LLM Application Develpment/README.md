# Table of Contents
1. [Introduction](#Introduction)
2. [Models, Prompts and Parsers](#Models,PromptsandParsers)
3. [Prompts](#Prompts)
## Introduction
LangChain - Open Source Framework for buildig LLM applications
Python and JS packages

## Models, Prompts and Parsers
- Models- LLM
- Prompts-style of input into the model
- parsers- parsing the response from the model; to do things downstream

### Using direct calls to OpenAI  
```
#!pip install openai
```
```
import os
import openai
```
```
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file
openai.api_key = os.environ['OPENAI_API_KEY']
```
```
def get_completion(prompt, model="gpt-3.5-turbo"):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0, 
    )
    return response.choices[0].message["content"]

prompt = f"""Translate the text \
that is delimited by triple backticks 
into a style that is {style}.
text: ```{customer_email}```
"""

print(prompt)
response = get_completion(prompt)
response
```


### Using LangChain
```
#!pip install --upgrade langchain
```

```
from langchain.chat_models import ChatOpenAI
```

```
# To control the randomness and creativity of the generated
# text by an LLM, use temperature = 0.0
chat = ChatOpenAI(temperature=0.0)
chat
```

### Prompts
- Prompt templates help reuse good prompt template for business applications that require larger and detailed prompts
- LangChain provides prompts for common operations like connecting to database
  Example like __Thought__, __Action__, __Observation__ as keywords for chain of thought reasoning.
  Thought-> is what the LLM should think. providing space for LLM helps improve the result
  Action-> keyword to carry out specific action
  Observation-> what it learnt from the action
  
```
template_string = """Translate the text \
that is delimited by triple backticks \
into a style that is {style}. \
text: ```{text}```
"""
```
```
from langchain.prompts import ChatPromptTemplate

prompt_template = ChatPromptTemplate.from_template(template_string)
prompt_template.messages[0].prompt.input_variables # style, text
```
```
customer_style = """American English \
in a calm and respectful tone
"""

customer_email = """
Arrr, I be fuming that me blender lid \
flew off and splattered me kitchen walls \
with smoothie! And to make matters worse, \
the warranty don't cover the cost of \
cleaning up me kitchen. I need yer help \
right now, matey!
"""

customer_messages = prompt_template.format_messages(
                    style=customer_style,
                    text=customer_email)

# Call the LLM to translate to the style of the customer message
customer_response = chat(customer_messages)
```
### Parsers





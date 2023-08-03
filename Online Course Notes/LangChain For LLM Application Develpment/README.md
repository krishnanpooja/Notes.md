# Table of Contents
1. [Introduction](#Introduction)
2. [Models, Prompts and Parsers](#Models,PromptsandParsers)
3. [Prompts](#Prompts)
4. [Parser](#Parsers)
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
Why do we need Parsers?
- Even if the output is as follows
  ```
  {
  "gift": False,
  "delivery_days": 5,
  "price_value": "pretty affordable!"
}
  
  ```
The type of this reponse is __str__
So response.content.get('gift') -> returns an ERROR

#### Parse the LLM output string into a Python dictionary
```
from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import Structured
```

```
gift_schema = ResponseSchema(name="gift",
                             description="Was the item purchased\
                             as a gift for someone else? \
                             Answer True if yes,\
                             False if not or unknown.")
delivery_days_schema = ResponseSchema(name="delivery_days",
                                      description="How many days\
                                      did it take for the product\
                                      to arrive? If this \
                                      information is not found,\
                                      output -1.")
price_value_schema = ResponseSchema(name="price_value",
                                    description="Extract any\
                                    sentences about the value or \
                                    price, and output them as a \
                                    comma separated Python list.")

response_schemas = [gift_schema, 
                    delivery_days_schema,
                    price_value_schema]
```

```
output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
format_instructions = output_parser.get_format_instructions()
```

```
review_template_2 = """\
For the following text, extract the following information:

gift: Was the item purchased as a gift for someone else? \
Answer True if yes, False if not or unknown.

delivery_days: How many days did it take for the product\
to arrive? If this information is not found, output -1.

price_value: Extract any sentences about the value or price,\
and output them as a comma separated Python list.

text: {text}

{format_instructions}
"""

prompt = ChatPromptTemplate.from_template(template=review_template_2)

messages = prompt.format_messages(text=customer_review, 
                                format_instructions=format_instructions)
response = chat(messages)
```

```
output_dict = output_parser.parse(response.content)
output_dict.get('delivery_days')
```

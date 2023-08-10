# Table of Contents
1. [Introduction](#Introduction)
2. [Models, Prompts and Parsers](#Models,PromptsandParsers)
3. [Prompts](#Prompts)
4. [Parser](#Parsers)
5. [Memory](#Memory)
6. [Chain](#Chain)
# Introduction
LangChain - Open Source Framework for building LLM applications
Python and JS packages

# Models, Prompts and Parsers
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
  "price_value": "pretty affordable!"}
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

# Memory
How do you remember previous part of the conversation

## ConversationBufferMemory
```
import os

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file

import warnings
warnings.filterwarnings('ignore')
```
```
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBuffer
```
```
llm = ChatOpenAI(temperature=0.0)
memory = ConversationBufferMemory()
conversation = ConversationChain(
    llm=llm, 
    memory = memory,
    verbose=True
)
```
```
conversation.predict(input="Hi, my name is Andrew")
```
__OUTPUT__
```
Entering new ConversationChain chain...
Prompt after formatting:
The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.

Current conversation:

Human: Hi, my name is Andrew
AI:

> Finished chain.
"Hello Andrew! It's nice to meet you. How can I assist you today?"
```

```
conversation.predict(input="What is 1+1?")
conversation.predict(input="What is my name?")
```
```
print(memory.buffer)
```
__OUTPUT__
```
print(memory.buffer)
print(memory.buffer)
Human: Hi, my name is Andrew
AI: Hello Andrew! It's nice to meet you. How can I assist you today?
Human: What is 1+1?
AI: 1+1 is equal to 2.
Human: What is my name?
AI: Your name is Andrew.
```

To add new things into the memory
```
memory.save_context({"input": "Hi"}, 
                    {"output": "What's up"})
```

LLM is stateless. Each Transaction is independent.
Chatbot appears to have memory by providing the full conversation as 'context'
As conversation becomes long, the process becomes expensive.

## ConversationBufferWindowMemory
```
from langchain.memory import ConversationBufferWindowMemory
memory = ConversationBufferWindowMemory(k=1) 
memory.save_context({"input": "Hi"},
                    {"output": "What's up"})
memory.save_context({"input": "Not much, just hanging"},
                    {"output": "Cool"})
memory.load_memory_variables({})
```
__OUTPUT__
```
{'history': 'Human: Not much, just hanging\nAI: Cool'}
```

## ConversationTokenBufferMemory

```
from langchain.memory import ConversationTokenBufferMemory
from langchain.llms import OpenAI
llm = ChatOpenAI(temperature=0.0)

memory = ConversationTokenBufferMemory(llm=llm, max_token_limit=30) # different LLMs have different way of counting tokens so we need to pass the llm
memory.save_context({"input": "AI is what?!"},
                    {"output": "Amazing!"})
memory.save_context({"input": "Backpropagation is what?"},
                    {"output": "Beautiful!"})
memory.save_context({"input": "Chatbots are what?"}, 
                    {"output": "Charming!"}

memory.load_memory_variables({})
```
__OUTPUT__
```
{'history': 'AI: Beautiful!\nHuman: Chatbots are what?\nAI: Charming!'}
```

## ConversationSummaryMemory
LLM writes summary of the conversation so far.
```
from langchain.memory import ConversationSummaryBufferMemory
# create a long string
schedule = "There is a meeting at 8am with your product team. \
You will need your powerpoint presentation prepared. \
9am-12pm have time to work on your LangChain \
project which will go quickly because Langchain is such a powerful tool. \
At Noon, lunch at the italian resturant with a customer who is driving \
from over an hour away to meet you to understand the latest in AI. \
Be sure to bring your laptop to show the latest LLM demo."

memory = ConversationSummaryBufferMemory(llm=llm, max_token_limit=100)
memory.save_context({"input": "Hello"}, {"output": "What's up"})
memory.save_context({"input": "Not much, just hanging"},
                    {"output": "Cool"})
memory.save_context({"input": "What is on the schedule today?"}, 
                    {"output": f"{schedule}"})
memory.load_memory_variables({})
```
__OUTPUT__: max token limit💯
```
{'history': 'System: The human and AI exchange greetings. The human asks about the schedule for the day. The AI provides a detailed schedule, including a meeting with the product team, work on the LangChain project, and a lunch meeting with a customer interested in AI. The AI emphasizes the importance of bringing a laptop to showcase the latest LLM demo during the lunch meeting.'}
```

### Additional Memory Types:
1. Vector data memory: Stores text in a vector DB and retreives the most relevant blocks of text
2. Entity memories: using LLM, it remembers details about specific entities
3. Combination of memories like conversation memory+entity
4. key:value format like in SQL


## Chain
   - LLMChain
   - Sequential Chains
        - SimpleSequentialChain
        - SequentialChain
    - Router Chain
### LLM Chain
Combination of LLM and prompt

```
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
```

```
llm = ChatOpenAI(temperature=0.9)
prompt = ChatPromptTemplate.from_template(
    "What is the best name to describe \
    a company that makes {product}?"
)
chain = LLMChain(llm=llm, prompt=prompt)
product = "Queen Size Sheet Set"
chain.run(product)
```
__OUTPUT__ : 
'The Bed Royale'

###  Sequential Chain
#### Simple Sequential Chain
```
from langchain.chains import SimpleSequentialChain
```
```
llm = ChatOpenAI(temperature=0.9)

# prompt template 1
first_prompt = ChatPromptTemplate.from_template(
    "What is the best name to describe \
    a company that makes {product}?"
)

# Chain 1
chain_one = LLMChain(llm=llm, prompt=first_prompt)
```
```
# prompt template 2
second_prompt = ChatPromptTemplate.from_template(
    "Write a 20 words description for the following \
    company:{company_name}"
)
# chain 2
chain_two = LLMChain(llm=llm, prompt=second_prompt)
```
```
overall_simple_chain = SimpleSequentialChain(chains=[chain_one, chain_two],
                                             verbose=True
                                            )
overall_simple_chain.run(product)
```
__OUTPUT__:
```
 Entering new SimpleSequentialChain chain...
Royal Comfort Linens
Royal Comfort Linens is a premium bedding company offering luxurious and comfortable linens for a peaceful and restful sleep.

> Finished chain.
'Royal Comfort Linens is a premium bedding company offering luxurious and comfortable linens for a peaceful and restful sleep.'
```
### Sequential Chain with multiple input and output
Multiple input and output:
Output of chain 1 is input to chain2
```
from langchain.chains import SequentialChain
```
```
llm = ChatOpenAI(temperature=0.9)

# prompt template 1: translate to english
first_prompt = ChatPromptTemplate.from_template(
    "Translate the following review to english:"
    "\n\n{Review}"
)
# chain 1: input= Review and output= English_Review
chain_one = LLMChain(llm=llm, prompt=first_prompt, 
                     output_key="English_Review"
                    )
```
```
second_prompt = ChatPromptTemplate.from_template(
    "Can you summarize the following review in 1 sentence:"
    "\n\n{English_Review}"
)
# chain 2: input= English_Review and output= summary
chain_two = LLMChain(llm=llm, prompt=second_prompt, 
                     output_key="summary"
                    )
```
```
# prompt template 3: translate to english
third_prompt = ChatPromptTemplate.from_template(
    "What language is the following review:\n\n{Review}"
)
# chain 3: input= Review and output= language
chain_three = LLMChain(llm=llm, prompt=third_prompt,
                       output_key="language"
                      )
```
```

# prompt template 4: follow up message
fourth_prompt = ChatPromptTemplate.from_template(
    "Write a follow up response to the following "
    "summary in the specified language:"
    "\n\nSummary: {summary}\n\nLanguage: {language}"
)
# chain 4: input= summary, language and output= followup_message
chain_four = LLMChain(llm=llm, prompt=fourth_prompt,
                      output_key="followup_message"
                     )
```
```
# overall_chain: input= Review 
# and output= English_Review,summary, followup_message
overall_chain = SequentialChain(
    chains=[chain_one, chain_two, chain_three, chain_four],
    input_variables=["Review"],
    output_variables=["English_Review", "summary","followup_message"],
    verbose=True
)
```
```
review = df.Review[5]
overall_chain(review)
```
__OUTPUT__:
```
> Entering new SequentialChain chain...

> Finished chain.
{'Review': "Je trouve le goût médiocre. La mousse ne tient pas, c'est bizarre. J'achète les mêmes dans le commerce et le goût est bien meilleur...\nVieux lot ou contrefaçon !?",
 'English_Review': "I find the taste mediocre. The foam does not hold, it's strange. I buy the same ones in stores and the taste is much better...\nOld batch or counterfeit!?",
 'summary': 'The reviewer is disappointed with the taste and foam quality of the product, suggesting a possible issue with the batch or authenticity.',
 'followup_message': "Réponse de suivi : Nous vous remercions d'avoir partagé votre avis concernant notre produit. Nous sommes sincèrement désolés d'apprendre que vous avez été déçu par le goût et la qualité de la mousse. Nous prenons vos commentaires au sérieux et nous tenons à vous assurer que la satisfaction de nos clients est notre priorité absolue. Afin de mieux comprendre ce qui s'est passé, nous procéderons à une vérification interne approfondie pour détecter toute éventuelle anomalie dans la production de cette série spécifique. Dans l'attente des résultats, nous vous invitons à contacter notre service clientèle qui se fera un plaisir de vous aider et de vous proposer une solution appropriée. Votre satisfaction est importante pour nous, et nous sommes déterminés à vous offrir une expérience positive avec nos produits. Merci pour votre confiance et votre patience."}
```

### Router Chain
Mulitple subchain each specializes for a particular input
<img width="447" alt="image" src="https://github.com/krishnanpooja/Notes.md/assets/8016149/0519118a-8b84-4ab3-8fbe-a22cbfeca8c5">

```
physics_template = """You are a very smart physics professor. \
You are great at answering questions about physics in a concise\
and easy to understand manner. \
When you don't know the answer to a question you admit\
that you don't know.

Here is a question:
{input}"""


math_template = """You are a very good mathematician. \
You are great at answering math questions. \
You are so good because you are able to break down \
hard problems into their component parts, 
answer the component parts, and then put them together\
to answer the broader question.

Here is a question:
{input}"""

history_template = """You are a very good historian. \
You have an excellent knowledge of and understanding of people,\
events and contexts from a range of historical periods. \
You have the ability to think, reflect, debate, discuss and \
evaluate the past. You have a respect for historical evidence\
and the ability to make use of it to support your explanations \
and judgements.

Here is a question:
{input}"""


computerscience_template = """ You are a successful computer scientist.\
You have a passion for creativity, collaboration,\
forward-thinking, confidence, strong problem-solving capabilities,\
understanding of theories and algorithms, and excellent communication \
skills. You are great at answering coding questions. \
You are so good because you know how to solve a problem by \
describing the solution in imperative steps \
that a machine can easily interpret and you know how to \
choose a solution that has a good balance between \
time complexity and space complexity. 

Here is a question:
{input}"""
```
```
prompt_infos = [
    {
        "name": "physics", 
        "description": "Good for answering questions about physics", 
        "prompt_template": physics_template
    },
    {
        "name": "math", 
        "description": "Good for answering math questions", 
        "prompt_template": math_template
    },
    {
        "name": "History", 
        "description": "Good for answering history questions", 
        "prompt_template": history_template
    },
    {
        "name": "computer science", 
        "description": "Good for answering computer science questions", 
        "prompt_template": computerscience_template
    }
]
```
```
from langchain.chains.router import MultiPromptChain
from langchain.chains.router.llm_router import LLMRouterChain,RouterOutputParser
from langchain.prompts import PromptTemplate
```
```
llm = ChatOpenAI(temperature=0)
```
```
destination_chains = {}
for p_info in prompt_infos:
    name = p_info["name"]
    prompt_template = p_info["prompt_template"]
    prompt = ChatPromptTemplate.from_template(template=prompt_template)
    chain = LLMChain(llm=llm, prompt=prompt)
    destination_chains[name] = chain  
    
destinations = [f"{p['name']}: {p['description']}" for p in prompt_infos]
destinations_str = "\n".join(destinations)
```
If the subject is not known use default LLM model to answer
```
default_prompt = ChatPromptTemplate.from_template("{input}")
default_chain = LLMChain(llm=llm, prompt=default_prompt)
```
```
MULTI_PROMPT_ROUTER_TEMPLATE = """Given a raw text input to a \
language model select the model prompt best suited for the input. \
You will be given the names of the available prompts and a \
description of what the prompt is best suited for. \
You may also revise the original input if you think that revising\
it will ultimately lead to a better response from the language model.

<< FORMATTING >>
Return a markdown code snippet with a JSON object formatted to look like:
```json
{{{{
    "destination": string \ name of the prompt to use or "DEFAULT"
    "next_inputs": string \ a potentially modified version of the original input
}}}}
```

REMEMBER: "destination" MUST be one of the candidate prompt \
names specified below OR it can be "DEFAULT" if the input is not\
well suited for any of the candidate prompts.
REMEMBER: "next_inputs" can just be the original input \
if you don't think any modifications are needed.

<< CANDIDATE PROMPTS >>
{destinations}

<< INPUT >>
{{input}}

<< OUTPUT (remember to include the ```json)>>"""
```
```
router_template = MULTI_PROMPT_ROUTER_TEMPLATE.format(
    destinations=destinations_str
)
router_prompt = PromptTemplate(
    template=router_template,
    input_variables=["input"],
    output_parser=RouterOutputParser(),
)

router_chain = LLMRouterChain.from_llm(llm, router_prompt)
```
```
chain = MultiPromptChain(router_chain=router_chain, 
                         destination_chains=destination_chains, 
                         default_chain=default_chain, verbose=True
                        )
chain.run("What is black body radiation?")
```
__OUTPUT__
```
{
    "destination": "physics",
    "next_inputs": "What is black body radiation?"
}
```

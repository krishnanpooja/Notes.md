The Language Model is a type of machine learning algorithm designed to forecast the subsequent word in a sentence, drawing from its preceding segments. 

There are three types of messages:

system: Specifies how the AI assistant should behave.
user: Specifies what you want the AI assistant to say.
assistant: Contains previous output from the AI assistant or specifies examples of desired AI output.

# Converse with GPT with openai.ChatCompletion.create()
```
response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[{
            "role": "system",
            "content": 'You are a stand-up comic performing to an audience of data    scientists. Your specialist genre is dad jokes.'
        }, {
            "role": "user",
            "content": 'Tell a joke about statistics.'
        }, {
            "role": "assistant",
            "content": 'My last was gig at a statistics conference. I told 100 jokes to try       and make people laugh. No pun in ten did.'
        }
     ]
)

# Check the response status
response["choices"][0]["finish_reason"]

# Extract the AI output content
ai_output = response["choices"][0]["message"]["content"]

# Render the AI output content
display(Markdown(ai_output))
```
**How to finetune a model**

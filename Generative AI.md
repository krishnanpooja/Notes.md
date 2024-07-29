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
Fine-tuning is the process of taking a pre-trained model and further training it on a domain-specific dataset.
1. choose a pretrained model and dataset
2. load the dataset
3. tokenizer
   ```
   tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
   ```

4. Initialize your base model
5. fine tune using Trainer method
    The Transformers library contains the Trainer class, which supports a wide range of training options and features such as logging, gradient accumulation, and mixed precision
 ```
   trainer = Trainer(
   model=model,
   args=training_args,
   train_dataset=small_train_dataset,
   eval_dataset=small_eval_dataset,
   compute_metrics=compute_metrics,

)

trainer.train()
```
6. Evaluate the model

**Difference between fine-tuning model and RAG**
1. Dynamic vs. Static: RAG excels in dynamic environments with up-to-date information, while fine-tuning may result in static models.
2. Model Customization: Fine-tuning offers more customization for writing style and behavior, while RAG focuses on information retrieva
3. Hallucinations: RAG is less prone to hallucinations, but fine-tuning can reduce hallucinations with domain-specific data.
4. Accuracy: Fine-tuning often provides higher accuracy for specialized tasks.
5. Transparency: RAG offers greater transparency in response generation.
6. Cost: RAG is generally more cost-effective than fine-tuning.
7. Complexity: Fine-tuning is more complex, requiring deeper knowledge of NLP and model training.

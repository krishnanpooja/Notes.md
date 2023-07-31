## Iterative Prompt Development
- you can improve the prompt in iteration like adding
  - "using at most 50 words" or "use at most 3 sentences" or "... characters" to shorten the result
  - Adding more specific info like "the description is intended for furniture retailers.. "
  - "At the end add product-ID in the technical specification"

##  Summarizing
Example Prompt:-
```
prompt = f"""
Your task is to generate a short summary of a product \
review from an ecommerce site. 

Summarize the review below, delimited by triple 
backticks, in at most 30 words. 

Review: ```{prod_review}```
"""

response = get_completion(prompt)
print(response)
```
- improve it further by adding more info like the end user using the resulting resposnse
  - giving instructions on topics to focus on "focusing on aspects that are relevant to the price and preceived  value"
  - Extract inforamtion rather than summarize - "Your task is to extract information... "
  - For summarizing a list of reviews:
    ```
    for i in range(len(reviews)):
    prompt = f"""
    Your task is to generate a short summary of a product \ 
    review from an ecommerce site. 

    Summarize the review below, delimited by triple \
    backticks in at most 20 words. 

    Review: ```{reviews[i]}```
    """

    response = get_completion(prompt)
    print(i, response, "\n")
   
    ```

## Inferring
Model takes text as input and performs some kind of analysis like extracting labels, sentiment analysis etc.
Example:
```
prompt = f"""
What is the sentiment of the following product review, 
which is delimited with triple backticks?

Review text: '''{lamp_review}'''
"""
response = get_completion(prompt)
print(response)
```
- improved by asking to answer in single word
- " Identify a list of emotions.."
-  Information extraction example:
```
prompt = f"""
Identify the following items from the review text: 
- Item purchased by reviewer
- Company that made the item

The review is delimited with triple backticks. \
Format your response as a JSON object with \
"Item" and "Brand" as the keys. 
If the information isn't present, use "unknown" \
as the value.
Make your response as short as possible.
  
Review text: '''{lamp_review}'''
"""
response = get_completion(prompt)
print(response)
```




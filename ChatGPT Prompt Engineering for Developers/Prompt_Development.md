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
  - 

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
- Inferring topic -"Determine 5 topics being discussed in the following text..."

## Transforming 
Transforming input to a different format
- Translating to different language
- input to HTML and output in JSON
```
user_messages = [
  "La performance du système est plus lente que d'habitude.",  # System performance is slower than normal         
  "Mi monitor tiene píxeles que no se iluminan.",              # My monitor has pixels that are not lighting
  "Il mio mouse non funziona",                                 # My mouse is not working
  "Mój klawisz Ctrl jest zepsuty",                             # My keyboard has a broken control key
  "我的屏幕在闪烁"                                               # My screen is flashing
]

for issue in user_messages:
    prompt = f"Tell me what language this is: ```{issue}```"
    lang = get_completion(prompt)
    print(f"Original message ({lang}): {issue}")

    prompt = f"""
    Translate the following  text to English \
    and Korean: ```{issue}```
    """
    response = get_completion(prompt)
    print(response, "\n")
```
__Output__
```
Original message (The language is French.): La performance du système est plus lente que d'habitude.
The performance of the system is slower than usual.

시스템의 성능이 평소보다 느립니다. 

Original message (The language is Spanish.): Mi monitor tiene píxeles que no se iluminan.
English: "My monitor has pixels that do not light up."

Korean: "내 모니터에는 밝아지지 않는 픽셀이 있습니다." 

Original message (The language is Italian.): Il mio mouse non funziona
English: "My mouse is not working."
Korean: "내 마우스가 작동하지 않습니다." 
...
```

- change the tone of the output
- correct Grammatical errors
  ```
    prompt = f"""Proofread and correct the following text
    and rewrite the corrected version. If you don't find
    and errors, just say "No errors found". Don't use 
    any punctuation around the text:
    ```{t}```"""
    response = get_completion(prompt)
    print(response)
  ```
output:
```
The girl with the black and white puppies has a ball.
No errors found.
No errors found.
There goes my freedom. They're going to bring their suitcases.
You're going to need your notebook.
That medicine affects my ability to sleep. Have you heard of the butterfly effect?
This phrase is to check chatGPT for spelling ability.
```


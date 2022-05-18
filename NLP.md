# Table of Contents
1. [RNN, LSTM](#LSTM)
1. [Transformers](#Transformers)



**Any typical NLP problem can be proceeded as follows:**

- Text gathering(web scraping or available datasets)
- Text cleaning(stemming, lemmatization)
- Feature generation(Bag of words) 
- Embedding and sentence representation(word2vec- CBOW, Skig gram)
- Training the model by leveraging neural nets or regression techniques
- Model evaluation
- Making adjustments to the model
- Deployment of the model.

**Parsing**

Parsing a document means to working out the grammatical structure of sentences, for instance, which groups of words go together (as “phrases”) and which words are the subject or object of a verb

- Tokenization- dividing a whole document into words.Tokens==words
- lemmatization- Remove ending of the words to retrieve base words like colors==color

**Stemming and Lemmatization**

Both generate the root form of the inflected words. The difference is that stem might not be an actual word whereas, lemma is an actual language word.

Stemming follows an algorithm with steps to perform on the words which makes it faster. Whereas, in lemmatization, you used WordNet corpus and a corpus for stop words as well to produce lemma which makes it slower than stemming. 


**Named Entity Recognition(NER)**

Recognize the Noun,Verb, adverb in the sentence

**Latent semantic indexing**
- Term-Document matrix and SVD
Latent semantic indexing is a mathematical technique to extract information from unstructured data. It is based on the principle that words used in the same context carry the same meaning.
In order to identify relevant (concept) components, or in other words, aims to group words into classes that represent concepts or semantic fields, this method applies Singular Value Decomposition  to the Term-Document matrix. As the name suggests this matrix consists of words as rows and document as columns.
LSI is computation heavy when compared to other models

**Latent semantic analysis**
Documents with words that are relevant in them

Steps:
- Large Corpus
- retrieve Term-Document Co-occurence matrix
- frequency transformations
- Apply cosine similarity to find semantic relatedness score

**TF-IDF**

TF-IDF- Provides information regarding how important a word is to a document in a collection
Term Frequency: is a scoring of the frequency of the word in the current document.
Inverse Document Frequency: is a scoring of how rare the word is across documents.

TF-IDF = TF*IDF
TF(t) = (Number of times term t appears in a document) / (Total number of terms in the document)
IDF(t)=log_e(Total number of documents / Number of documents with term t in it).

**Dependency Parsing/Syntactic Parsing**

Analyzes the grammatical structure of a sentence, establishing relationships between "head" words and words which modify those heads.like from the word moving to the word faster indicates that faster modifies moving.

**GloVe**

Coined from Global Vectors, is a model for distributed word representation. The model is an unsupervised learning algorithm for obtaining vector representations for words. This is achieved by mapping words into a meaningful space where the distance between words is related to semantic similarity.

http://text2vec.org/glove.html

THe GloVe algorithm consists of following steps:
Collect word co-occurence statistics in a form of word co-ocurrence matrix X. Each element Xij of such matrix represents how often word i appears in context of word j. Usually we scan our corpus in the following manner: for each term we look for context terms within some area defined by a window_size before the term and a window_size after the term. Also we give less weight for more distant words, usually using this formula:
decay=1/offset

**Topic modeling**

Its a type of statistical modeling for discovering the abstract “topics” that occur in a collection of documents. Latent Dirichlet Allocation (LDA) is an example of topic model and is used to classify text in a document to a particular topic

**Statistical Language Modeling**

Language Modeling and LM for short, is the development of probabilistic models that are able to predict the next word in the sequence given the words that precede it.

**Conditional Random Fields**

Conditional Random Fields is a class of discriminative models best suited to prediction tasks where contextual information or state of the neighbors affect the current prediction. CRFs find their applications in named entity recognition, part of speech tagging, gene prediction, noise reduction and object detection problems

**POS Tagging**

n-gram approach, referring to the fact that the best tag for a given word is determined by the probability that it occurs with the n previous tags. This approach makes much more sense than the one defined before, because it considers the tags for individual words based on context
In the part of speech tagging problem, the observations are the words themselves in the given sequence.

As for the states, which are hidden, these would be the POS tags for the words.

The transition probabilities would be somewhat like P(VP | NP) that is, what is the probability of the current word having a tag of Verb Phrase given that the previous tag was a Noun Phrase.

Emission probabilities would be P(john | NP) or P(will | VP) that is, what is the probability that the word is, say, John given that the tag is a Noun Phrase.
Apply viterbi algorithm to this

**Seq2Seq**

A RNN layer (or stack thereof) acts as "encoder": it processes the input sequence and returns its own internal state. Note that we discard the outputs of the encoder RNN, only recovering the state. This state will serve as the "context", or "conditioning", of the decoder in the next step.

Another RNN layer (or stack thereof) acts as "decoder": it is trained to predict the next characters of the target sequence, given previous characters of the target sequence. Specifically, it is trained to turn the target sequences into the same sequences but offset by one timestep in the future, a training process called "teacher forcing" in this context. Importantly, the encoder uses as initial state the state vectors from the encoder, which is how the decoder obtains information about what it is supposed to generate. Effectively, the decoder learns to generate targets[t+1...] given targets[...t], conditioned on the input sequence.

The translated output of the previous layer is fed into the current along the input vector

**Stop List**

The list of words that are not to be added is called a stop list. Stop words are deemed irrelevant for searching purposes because they occur frequently in the language for which the indexing engine has been tuned.

**CBOW and Skip-gram**

In the CBOW model, the distributed representations of context (or surrounding words) are combined to predict the word in the middle. While in the Skip-gram model, the distributed representation of the input word is used to predict the context.

- Skip-gram

The dimensions of the input vector will be 1xV — where V is the number of words in the vocabulary — i.e one-hot representation of the word. The single hidden layer will have dimension VxE, where E is the size of the word embedding and is a hyper-parameter. The output from the hidden layer would be of the dimension 1xE, which we will feed into an softmax layer. The dimensions of the output layer will be 1xV, where each value in the vector will be the probability score of the target word at that position.
According to our earlier example if we have a vector [0.2, 0.1, 0.3, 0.4], the probability of the word being mango is 0.2, strawberry is 0.1, city is 0.3 and Delhi is 0.4.
The back propagation for training samples corresponding to a source word is done in one back pass. So for juice, we will complete the forward pass for all 4 target words ( have, orange, and, eggs). We will then calculate the errors vectors[1xV dimension] corresponding to each target word. We will now have 4 1xV error vectors and will perform an element-wise sum to get a 1xV vector. The weights of the hidden layer will be updated based on this cumulative 1xV error vector.

- CBOW

The fake task in CBOW is somewhat similar to Skip-gram, in the sense that we still take a pair of words and teach the model that they co-occur but instead of adding the errors we add the input words for the same target word.

The dimension of our hidden layer and output layer will remain the same. Only the dimension of our input layer and the calculation of hidden layer activations will change, if we have 4 context words for a single target word, we will have 4 1xV input vectors. Each will be multiplied with the VxE hidden layer returning 1xE vectors. All 4 1xE vectors will be averaged element-wise to obtain the final activation which then will be fed into the softmax layer.

Skip-gram: works well with a small amount of the training data, represents well even rare words or phrases.

CBOW: several times faster to train than the skip-gram, slightly better accuracy for the frequent words.

one-hot vector*embedding matrix = embedding vector for the word
(this is fed into RNN for sentiment classification)

**Text extraction and mining:**
The process of extracting raw text from the input data by getting rid of all the other non-textual information, such as markup, metadata, etc., and converting the text to the required encoding format is called text extraction and cleanup.
Following are the common ways used for Text Extraction in NLP:

	1.Named Entity Recognition
	2.Sentiment Analysis
	3.Text Summarization
	4.Aspect Mining
	5.Topic Modeling

**What is the meaning of Text Normalization in NLP?**

Consider a situation in which we’re operating with a set of social media posts to find information events. Social media textual content may be very exceptional from the language we’d see in, say, newspapers. A phrase may be spelt in multiple ways, such as in shortened forms, (for instance, with and without hyphens), names are usually in lowercase, and so on. When we're developing NLP tools to work with such kinds of data, it’s beneficial to attain a canonical representation of textual content that captures these kinds of variations into one representation. This is referred to as text normalization

**What is an ensemble method in NLP?**

An ensemble approach is a methodology that derives an output or makes predictions by combining numerous independent similar or distinct models/weak learners. An ensemble can also be created by combining various models such as random forest, SVM, and logistic regression.
Bias, variance, and noise, as we all know, have a negative impact on the mistakes and predictions of any machine learning model. Ensemble approaches are employed to overcome these drawbacks


<a name="LSTM"></a>
## RNN LSTM GRU

**What is RNN? LSTM and GRU?**

vanilla RNN,
long short-term memory (LSTM), proposed by Hochreiter and Schmidhuber in 1997, and
gated recurrent units (GRU), proposed by Cho et. al in 2014.

![image](https://user-images.githubusercontent.com/8016149/167280572-f3da6d2b-82ef-4822-a942-b7bc72542e21.png)

***What is vanishing gradient problem?***

Ans: In RNN to train the network you backpropagate through time, at each step the gradient is calculated. The gradient is used to update weights in the network. If the effect of the previous layer on the current layer is small then the gradient value will be small and vice-versa. If the gradient of the previous layer is smaller then the gradient of the current layer will be even smaller. This makes the gradients exponentially shrink down as we backpropagate. Smaller gradient means it will not affect the weight updation. Due to this, the network does not learn the effect of earlier inputs. Thus, causing the short-term memory problem.
LSTM
![image](https://user-images.githubusercontent.com/8016149/167283059-129c312d-1115-40f2-a2e6-40a15ce68125.png)


<a name="Transformers"></a>
## Transformers
https://www.tensorflow.org/text/tutorials/transformer

LSTM is slower and sequentially processes data. Transformers process data in parallel.
It makes no assumptions about the temporal/spatial relationships across the data. This is ideal for processing a set of objects (for example, StarCraft units).
Layer outputs can be calculated in parallel, instead of a series like an RNN.
Distant items can affect each other's output without passing through many RNN-steps, or convolution layers (see Scene Memory Transformer for example).
It can learn long-range dependencies. This is a challenge in many sequence tasks.

The downsides of this architecture are:

For a time-series, the output for a time-step is calculated from the entire history instead of only the inputs and current hidden-state. This may be less efficient.
If the input does have a temporal/spatial relationship, like text, some positional encoding must be added or the model will effectively see a bag of words.

The Transformer in NLP is a novel architecture that aims to solve sequence-to-sequence tasks while handling long-range dependencies with ease. It relies entirely on self-attention to compute representations of its input and output WITHOUT using sequence-aligned RNNs or convolution.

**Self-Attention**

the ability to attend to different positions of the input sequence to compute a representation of that sequence.
The three kinds of Attention possible in a model:
1.Encoder-Decoder Attention: Attention between the input sequence and the output sequence.
2.Self attention in the input sequence: Attends to all the words in the input sequence.
3.Self attention in the output sequence: One thing we should be wary of here is that the scope of self attention is limited to the words that occur before a given word. This prevents any information leaks during the training of the model. This is done by masking the words that occur after it for each step. So for step 1, only the first word of the output sequence is NOT masked, for step 2, the first two words are NOT masked and so on.

**Query, Key and Value**

1. Query Vector: q= X * Wq. Think of this as the current word.
2. Key Vector: k= X * Wk. Think of this as an indexing mechanism for Value vector. Similar to how we have key-value pairs in hash maps, where keys are used to uniquely index the values.
3.Value Vector: v= X * Wv. Think of this as the information in the input word.

dk = tf.cast(tf.shape(k)[-1], tf.float32)
For example, consider that Q and K have a mean of 0 and variance of 1. Their matrix multiplication will have a mean of 0 and variance of dk. So the square root of dk is used for scaling, so you get a consistent variance regardless of the value of dk. If the variance is too low the output may be too flat to optimize effectively. If the variance is too high the softmax may saturate at initialization making it difficult to learn.

What we want to do is take query q and find the most similar key k, by doing a dot product for q and k. The closest query-key product will have the highest value, followed by a softmax that will drive the q.k with smaller values close to 0 and q.k with larger values towards 1. This softmax distribution is multiplied with v. The value vectors multiplied with ~1 will get more attention while the ones ~0 will get less. The sizes of these q, k and v vectors are referred to as “hidden size” by various implementation

![image](https://user-images.githubusercontent.com/8016149/167283288-558f09c5-2218-417e-bd3d-a2da4d514d37.png)

Then divide this product by the square root of the dimension of key vector.
This step is done for better gradient flow which is specially important in cases when the value of the dot product in previous step is too big. As using them directly might push the softmax into regions with very little gradient flow.

#Transformers
*The Transformer* 
The input sentence is passed through N encoder layers that generates an output for each token in the sequence.
The decoder attends to the encoder's output and its own input (self-attention) to predict the next word.
1. **Encoder**

It takes in the input sequence (x1.. xn) parallely.
Each encoder has two sub-layers.

1.A multi-head self attention mechanism on the input vectors (Think parallelized and efficient sibling of self attention).
2. A simple, position-wise fully connected feed-forward network (Think post-processing).

2. **Decoder**

It takes in the output sequence (y1..yn) paralley (thanks to multi-head attention)
Each decoder has three sub-layers.

1. A masked multi-head self attention mechanism on the output vectors of the previous iteration.
2. A multi-head attention mechanism on the output from encoder and masked multi-headed attention in decoder.
3. A simple, position-wise fully connected feed-forward network (think post-processing).

![image](https://user-images.githubusercontent.com/8016149/167283445-34ce9e7a-e1e8-417f-b54c-6d3f3a3ad190.png)

![image](https://user-images.githubusercontent.com/8016149/167283613-297c469e-7d01-4e42-b822-ac83433bc043.png)

**Input to transformers**

This injected vector is called “positional encoding” and are added to the input embeddings at the bottoms of both encoder and decoder stacks.
Decoder: This positional encoding + word embedding combo is then fed into a masked multi-headed self attention.
The outputs from the encoder stack are then used as multiple sets of key vectors k and value vectors v, for the “encoder decoder attention”
The q vector comes from the “output self attention” layer.

**complexity of attention matrix in transformer?**

When the original Attention paper was first introduced, it didn't require to calculate Q , V and K matrices, as the values were taken directly from the hidden states of the RNNs, and thus the complexity of Attention layer is O(n^2·d) 

## Optimization

1. Size reduction- Easier to download, store and less memory usage
2. latency reduction- amt of time taken for a single inference
3. Accelrator compatiblity - run on TPUs

Optimization Techniques using Tensorflow Lite - Quantization, Pruning, Clustering

**what is quantization? ways to do this?**

Quantization works by reducing the precision of the numbers used to represent a model's parameters, which by default are 32-bit floating point numbers. This results in a smaller model size and faster computation.

** WER **

Basically, WER is the number of errors divided by the total words. To get the WER, start by adding up the substitutions, insertions, and deletions that occur in a sequence of recognized words. Divide that number by the total number of words originally spoken. The result is the WER.

https://keras.io/examples/vision/knowledge_distillation/

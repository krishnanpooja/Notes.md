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

**What is RNN? LSTM and GRU?**
vanilla RNN,
long short-term memory (LSTM), proposed by Hochreiter and Schmidhuber in 1997, and
gated recurrent units (GRU), proposed by Cho et. al in 2014.

![image](https://user-images.githubusercontent.com/8016149/167280572-f3da6d2b-82ef-4822-a942-b7bc72542e21.png)



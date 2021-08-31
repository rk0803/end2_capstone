## BACKGROUND
The AI Community, facing one of the most challenging tasks is to be able to train computers to think, read and write like human beings. This is what is called as Natural Language Processing, i.e to be able to process the natural language (rather than just programming language), just like humans do thereby trying to bridge the gap between humans and computers. These tasks can range from filling in the blanks, finding(classifying) the sentiments from plethora of text available in the form of reviews, tweets etc., giving relevant answers from world of information available out there (open domain) or from a specific domain (closed domain), etc. to summarizing the text etc. The list is unending. The objective is to be able to process the loads of information out there without getting overwhelmed and inundated, to be able to interact with computers without needing to learn anything specific(like a programming language). It should just as natural and close as interacting with another human being.

The research and consequently development of tricks/techniques in this domain once focused on handling for specific tasks with specific models/frameworks etc. Today, (during the past two years mostly) powerful general-purpose language models are mushrooming which can be fine-tuned for a wide variety of different tasks. While promising, efforts to this point have applied these general-purpose models to tasks (such as sentiment analysis) for which a human could produce the solution without additional background knowledge. This is helpful in multiple ways. While a model with billions of parameters, can be pre-trained on a large (obscene) amount of data to ensure better learning, which can take huge amount of resources, it can be tweaked to a specific data/ purpose with 'handlable' amount of data and resources. 

Hence the ensuing research, along with developing new and better frameworks, is also on the lines of how to 'tame' these models for specific tasks etc. This leads us to this capstone project as part of the END2 course at 'The School of AI'. The purpose of this capstone project is to build a closed domain question answering system.

## Preamble(for capstone project)

We need three components to achieve this:

A (pre-trained) generator model like **BERT**.
A (pre-trained) retriever model like **DPR**.
An indexed KB of text documents (JSON, our dataset)
This is how our architecture would look like:

![image](https://user-images.githubusercontent.com/82941475/131240796-8ab81846-de2a-466c-9463-530a6b47a9e3.png)


we need to make our own bi-encoder and minimize this equation <img src="https://render.githubusercontent.com/render/math?math=p(z|x) \propto e^{d(z)^Tq(x)}">

Where  **d(z)** is the encoded document vector (i.e. CLS Token) coming from a pre-trained BERT1 model (never to be trained), **q(x)** is the encoded question vector (i.e. CLS Token) coming from a pre-trained BERT2 model (to be trained (fine-tuned)). We would not be training the Document encoder, this is to ensure that if new data is added to the system, there is no need to retrain it. But the question encoder needs to be trained, so as to minimize the loss and overall help in retrieval.
Then we train one BERT model (fine-tuning a pre-trained BERT model) such that log likelihood of **p(z|x)** is minimized, i.e. siamese network, i.e. similarity network!

Once the bi-encoder is ready to an acceptable level, all of JSON passages (Z) are through the document encoder and an index of the document vectors is stored.On the inferencing mode, **FAISS (Facebook AI Similarity Search)** is implemented to send the top k (where k is a hyper-parameter, required to be set) documents to the next step.</br>

After that pre-trained seq2seq (BART) model is taken (trained as denoising auto-encoder). The the top k documents retrieved by FAISS are concatenated with the query (raw text) and fed to the BART encoder (The process can be started some pre-trained document summarization BART for both encoder-decoder as the starting weights). This is done to collect answers from multiple documents if required. The RAW TEXT of each document needs to be concatenated, pre-pended by the Question Query. The decoder then predict y, the answer!
### Points to be taken care of:
1. The decoder can be trained EITHER to predict the exact sentence wanted, OR Some other sentence that is semantically similar to yours (although second one is desirable, and compare them using the document encoder outputs)
2. If the answer contains code, it should be ensured to predict exactly the same code. 

### Additional training notes provided:

- In our dataset, we are sort of writing our own answer, and exact wording may not be available in the document (z). It should then be decided as to how to then compare actual (**y_{ground_truth}**) with the prediction (**y_{predicted}**).  Implementing teacher forcing on the final decoder is not expected. Preferably, keep an encoded vector for each of the **y_{gound_truth}**, send the **y_{predicted}** to the original BERT1 Document Encoder and then compare it (loss function). Decide on some semantic loss that allows the model to predict sentences that are not word to word same as the ground_truth. 
- Taking a RAG model or existing DPR model and train it on the dataset of this project is not allowed.
- The training process needs to be document, and separate (properly named) notebooks be linked in the readme file along with notes to be able to understand how the model is trained.
- The training logs MUST be available for review.
- The objective is to solve the problem end to end. MUST have trained for over 500 EPOCHS in total (for all the models combined) and that the loss is reduced substantially from the starting point should be visible
- The dataset must be split into 80/20 and the test accuracy must be highlighted in the readme file.The results on 100 randomly picked samples from the test dataset and the results must be shown in the following format: 	Question, 	Prediction, 	Exact Answer, 	Document

# Architecture of the (proposed) Model
The complete model is mix of a BERT (**Bidirectional Encoder Representations from Transformers**), and BART (**Bidirectional and Auto-Regressive Transformer**)
The overview of the architecture would be like this (while we zoom into the details gradually):
![image](https://user-images.githubusercontent.com/82941475/131285918-9ebc13e1-4ff1-49af-9061-d14c5f1a9f84.png)

So as it can be seen, that there are two major modules in this model: Retriever and Generator. Lets first zoom into Retriever.
### Retriever:
Retriever encodes both the query and documents. So we have something like bi-encoder model which encodes the queries and documents, passes it to the similarity search, and trains the bi-encoder to retrieve top k (k to be provided) matching documents z, given the query x such that the probability p(z|x) is maximized.  This can be done through different appraoches. (Details are provided in implementation details.)
These top k documents are then passed on to the generator model.

### Generator

Generator, after getting the documents, along with the query, generates the answer to the query, maximizing the probability p(y|x,z) or minimizing the log-likelihood of this probability. 
 

**Retriever** | **Generator**
----------|---------
![image](https://user-images.githubusercontent.com/82941475/131285968-0cf16bf4-20f8-4ff5-a81b-5455ef702661.png) | ![image](https://user-images.githubusercontent.com/82941475/131286003-0b31e7f6-b3ca-4824-a020-de858564c5ff.png)


Retriever is like a similarity network (or siamese network), where we are providing it with documents and queries, it retrieves the most relevant documents to the query. 

![image](https://user-images.githubusercontent.com/82941475/131241319-e1062970-fe61-48f0-84ea-a3a94eb9989a.png)

 ### Archtectural Details: [Justifications for each point is provided next]
 1. Bi-encoder uses two encoders, one for the query and one for the documents. We use BERT_{1} pre-trained model for query,  and a second BERT_{2} pre-trained model for document encoding. We fine-tune/update BERT_{1} encoder during the training process to facilitate the rerieval process i.e. maximizing the probability p(z|x) or minimizing the log-likelihood -log(z|x), for each z retrieved. We DO NOT update/fine-tune the document encoder BERT_{2} during training. Updating BERT_{2} also would mean continually updating the index as well, it becomes computationally expensive operation. Additionally, if we want to add more documents, we would need to re-train the whole document encoder again, another computationally intensive work.
 2. Similarity search here can be cosine similarity or any other vector similarity measure. But this again becomes compute intensive as similarity of the query vector with each of the document is calculated, ranked in decreasing order, and then top k are selected. Instead we use FAISS (Facebook AI Similarity Search), which is much faster. 
 3. BART is used as the generator model. This takes as input the documents (passed on from retriever) concatenated together, pre-pended with the query, generates the answer token by token, minimizing the log-likelihood of p(y|x,z). So this BART generator works as described below: 
So e.g. three documents, z1, z2 and z3 get selected given the query x, maximizing the probability p(z|x). So now we have three latent documents for the query x : (x,z1), (x,z2) and (x,z3). Now, lets say BART model generates sequence for each of the latent documents, say, y11, y12 for z1, y21, y22 for z2 and y31, y32 for z3. Now we have 6 hypotheses to test. So we calculate the probability for each of these for each of the documents. So for example, for  y11, <img src="https://render.githubusercontent.com/render/math?math=p(y11|x)= \Sigma_{i=1}^3 p(y11|x,zi) \times  p(zi|x)  "> This is done for each yij, i=1,2,3 and j=1,2 to get the probability p(yij|x). Maximum value of this is returned.
This whole process is depicted in the image below:

![image](https://user-images.githubusercontent.com/82941475/131468248-9338f840-779f-442c-b2a5-589af548fc51.png)

## Difference from the RAG paper:
In the RAG paper, the authors/developers of the model jointly train the retriever and generator components without any direct supervision on what
document should be retrieved. Given a fine-tuning training corpus of input/output pairs (xj ; yj), minimizing the negative marginal log-likelihood of each target. 

In the proposed model, we are going incrementally. We first train the bi-encoder. Once it is encoded to sufficiently desired level of accuracy, it is put to use to select the top k documents given the query. These top k documents are then passed on the generator to select the one with maximum p(y|x,z).

## Justifications:
### Why Bert:
![image](https://user-images.githubusercontent.com/82941475/131286422-b52a97d8-3716-44fb-88b5-bbe6f174906d.png)

BERT is a transformers model pretrained on a large corpus of English data in a self-supervised fashion. This means it was pretrained on the raw texts only, with no humans labelling involved. More precisely, it was pretrained with two objectives:

**Masked language modeling (MLM)**: taking a sentence, the model randomly masks 15% of the words in the input then run the entire masked sentence through the model and has to predict the masked words. It allows the model to learn a bidirectional representation of the sentence.

**Next sentence prediction (NSP)**: the models concatenates two masked sentences as inputs during pretraining. Sometimes they correspond to sentences that were next to each other in the original text, sometimes not. The model then has to predict if the two sentences were following each other or not.
**MLM**| **NSP**
-----|-----
![image](https://user-images.githubusercontent.com/82941475/131286537-817a0e82-10b9-4bc1-8626-c87a5008a287.png) | ![image](https://user-images.githubusercontent.com/82941475/131241384-2c43dfe7-2103-4a32-a4b5-d2c9ab31c16c.png)

Thus, the model can learn an inner representation of the English language that can then be used to extract features useful for downstream tasks, i.e. if you have a dataset of sentences (queries or documents in our case), you can use the features produced by the BERT model as inputs to further stages as required.

### Why FAISS,  and what is FAISS afterall?
Faiss is a library — developed by Facebook AI — that enables efficient similarity search.

So, given a set of vectors, we can index them using Faiss — then using another vector (the query vector), we search for the most similar vectors within the index.

Now, Faiss not only allows us to build an index and search — but it also speeds up search times to ludicrous performance levels. 
Lets first what all indexing options are available and what they do.
a) IndexFlatL2 measures the L2 (or Euclidean) distance between all given points between our query vector, and the vectors loaded into the index. It’s simple, very accurate, but not too fast.

![image](https://user-images.githubusercontent.com/82941475/131287827-e456b25d-04cd-462f-93df-903e16de3692.png)

Using the IndexFlatL2 index alone is computationally expensive, it doesn’t scale well.

When using this index, we are performing an exhaustive search — meaning we compare our query vector xq to every other vector in our index. Our index quickly becomes too slow to be useful as the dataset size increases.

Faiss allows us to add multiple steps that can optimize our search using many different methods. A popular approach is to partition the index into Voronoi cells.

![image](https://user-images.githubusercontent.com/82941475/131287906-648ec5ae-7c48-4cfb-b9c2-8ca108817c9e.png)

Using this method, we would take a query vector xq, identify the cell it belongs to, and then use our IndexFlatL2 (or another metric) to search between the query vector and all other vectors belonging to that specific cell.

So, we are reducing the scope of our search, producing an approximate answer, rather than exact (as produced through exhaustive search).
This is just a glimpse of we can reduce the search time (though at some cost of getting an approximate answer), there are multiple variations available for key optimizations etc. with varieity of different indexing techniques. These can be referred to here https://www.pinecone.io/learn/faiss-tutorial/.
 A lot can be done using these different  indexes, and each has many parameters that can be fine-tuned to our specific accuracy/speed requirements and we can produce some truly impressive results, at lightning-fast speeds very easily with Faiss.

### why BART
BART is a denoising autoencoder for pretraining sequence-to-sequence models. BART is trained by (1) corrupting text with an arbitrary noising function, and (2) learning a model to reconstruct the original text.

BART is pre-trained by minimizing the cross-entropy loss between the decoder output and the original sequence.
We have seen that BERT is a bidirectional encoder, that it can see the full sequence before making a prediction.
Auto regressive Models used for text generation, such as GPT2, are pre-trained to predict the next token given the previous sequence of tokens.

BART has both an encoder (like BERT) and a decoder (like GPT), essentially getting the best of both worlds.
The encoder uses a denoising objective similar to BERT while the decoder attempts to reproduce the original sequence (autoencoder), token by token, using the previous (uncorrupted) tokens and the output from the encoder.

![image](https://user-images.githubusercontent.com/82941475/131287998-71bafc00-c36a-483c-b4a7-1676ce43f7e5.png)

BART is particularly effective when fine tuned for text generation. 


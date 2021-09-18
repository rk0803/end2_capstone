# Capstone Project Part 3
This part of the project builds, trains a model to answer the questions on pytorch.
It was expected to be able to handle pytorch from all angles and sources, viz., Github, pytorch documentation, youtube etc. and model should be trained on comprehensive dataset created collabortaively.
However, I have taken the dataset from my team (pytorch documentation ) only. This was mainly due to two reasons:
1) I was thoroughly familiar with the nature of the data
2) More importantly, even on colab pro, this small dataset(approximatrly 18500) records, was causing system crash. At some points, I had to divide even this small set into smaller subsets.

The flow of the work is like as shown below:
![image](https://user-images.githubusercontent.com/82941475/133426875-7cc55a13-d193-4bcb-8f5e-3e65985c73dd.png)
1) **data_preparation.ipynb** prepares the data. It reads in a json file, coverts it into csv file, splits it into test and train set and writes back the train_qa.csv, and test_qa.csv. 
2) **document_encoder.ipynb** encodes the documents, using BERT model. This BERT encoder is never trained and only pre-trained embeddings were taken for the documents.
3) **question_encoder.ipynb** encodes the queries. This BERT encoder is trained using the corresponding context of the query to calculate the loss function. The loss function is -ve log of the dot product ot the query embedding with the document embedding. The encoder tries to minimize that.
4) **DPR_module.ipynb** This module works to retrieve the top k documents relevant to the query once the query encoder is satisfactorily trained. FAISS is used here to retrieve the relevant documents. The program then goes on to create a new jason file which has the query and top k documents as the input_text and corresponding answer as the out_text.
5) **gen_BART.ipynb** This module reads this new dataset and tries to train the answer generator using BART.
Once generator module is trained, it is fed the questions as input and answers are predicted. These predicted answers are then compared with actual answers and bertscore is calculated.

## Key Implementation Details to be noted
1) For both question and document encoder, *BertModel* and *BertTokenizer* were used. \[CLS\] token from the embedding layer is pickedup. 
Document encoder is never trained. So the output is picked up, and \[CLS\] token of dimension 768 is taken and tensor of size \[no_of_training_examples, 768\] is created. Since CUDA was running out of memory, this was done in two parts. So encoded1_z.pt, encoded2_z.pt were created. encoded3_z.pt is encodings for test set.
For Question Encoder:
A TensorDatset of encodings of documents were created to be able to load this batch-wise. 
A SequentialSampler is used to load both question samples and document encoding samples.
\[CLS\] token of dimension 768 is taken from the output of question encoder model. This is a tensor of size \[batch_size, 768\]. 
Negative Log of dot product of question and document encoding is the loss, which is back propagated to train the model. The loss reported in the notbook is total loss for 14011 samples. The average loss can be obtained from there. The averge training loss reported is a reported below for some epochs
epoch | Training loss | Test loss
------|-----------|------
1  | 0.2588	|
13| 0.129994 | 0.3999
23| 0.095982 | 0.4203


Optimizer used is AdamW with learning rate 5e-5.
The model was trained for 23 (in steps of 3 + 10 + another 10)epochs. This is not a magic number but a careful training so that CUDA doesnt run of memory. 
The training loss steadily decreased with few bumps on the way as can be seen in [img]question_training_loss.
After every 10 epochs, the model was evaluated on test set, to evaluate overfitting.
The model was saved(best_bert_q.pt).
This trained model was then passed to DPR_module to be retrieve top k relevant documents (k was kept to be 3, with no particular reason but a starting number to be played around with, but later on taken as it is.)
DPR Module
Here we take already trained question encoder, and document embeddings.
Use FAISS as the similarity module as this is more optimized than calculating the cosine similarity of each document with a given query.
After loading the document embeddings in a tensor an index needs to be created. There are varieties of indexes available. I have chosen IndexFlatL1 to start with. This is another aspect which can be played around with, which was the idea initially, but later on this was kept as it is.
The FAISS index expects a numpy array. So document encodings were converted into numpy array before building the index.
The query encodings from the trained enocder were taken, converted into numpy array and passed to the index to return the top k documents similar to query.
Index returns the distance and document indexes. These indexes were taken and those documents picked up from documents dataset.
A new json file with query+document(1)+document(2)+document(3) as in_text, and answer to the query as out_text was created to be passed on BART generator.(newqa_train.json, newqa_test.json)
BART Generator
BartTokenizer and BartForConditionalGeneration with 'facebook/bart-large' was used.
This new dataset is read, tokenized and a dataset for both training (train_dataset) and evaluation (test_dataset) were created. 
Important point to note here is to makes sure that the decoder inputs are correctly shifted and still include the padding tokens while in the labels the padding tokens are replaced by -100 such that they are ignored in the model loss. *shift_tokens_right* takes care of this aspect.
The model was trained and the loss steadily goes down as can be seen below ![image]Bart_loss.png
Overall training reported was **3.02092201767093**. The other related metrics reported are in the notbook gen_BART.ipynb. The model reported a test loss of **2.4269748294**, confirming that the model is not overfitting to the training data.
This model was then saved in trained_BART, with config.jason, pytorch_model.bin and all other related files.
This model was then used for generating answers for the given questions. 100 samples from test_dataset were taken for this purpose. 
A review of predictions (predictions_1.txt) revealed a poor quality predictions. So bertscore was not calculated.
As can be seen from the predictions, the model is way below expected performance. For each of the question, answer generated is just one word. This could be due to undertraining of 
a) question encoder. Cleary if the question encoder was better trained (for 100 or more epochs), question encodings could have been better leading to better match of the documents retrieved from FAISS.
b) undertraining of BART generator. If resources and could have permitted, the model could be trained for 2 or 3 epochs. (each epoch takes arounf 3 hours and more than the disk space available in colab pro. Some of the check-points were actually deleted while to training was going on to create space on the disk.)
c) Both the question encoder, BART generator together can be trained better for this overall work to be put to some use.
d) A better dataset

Key Takeaway:
Complete framework is ready and flow explained in the [image]. The framework can be used on a different dataset and evaluted and also can be further trained.

Key Learnings
Lots of hits and trials and head scratching happened in the beginning with things slowly settling down and fog being lifted.
1) First and most important, to be able to research and play around in the NLP arena, resources must be backed up by a large corporation with dedicated funds. At the individual level, with even colab, it was not an enough. This sort of comfirmed the statement the NLP is not for poor people.
2) Various concepts were cleared and how to read the model details and to be able to utilize the relevant return field values was one major accomplishment for me.
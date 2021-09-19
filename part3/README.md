# Capstone Project Part 3
This part, in continuation with part 2, of the project builds, trains a model to answer the questions on pytorch.
It was expected to be able to handle pytorch from all angles and sources, viz., Github, pytorch documentation, youtube etc. and model should be trained on comprehensive dataset created collabortaively.
However, I have taken the dataset from my team (pytorch documentation ) only. This was mainly due to two reasons:
1) I was thoroughly familiar with the nature of the data
2) More importantly, even on colab pro, this small dataset(approximatrly 18500) records, was causing system crash. At some points, I had to divide even this small set into smaller subsets.

The flow of the work is as shown below:
![image](https://user-images.githubusercontent.com/82941475/133426875-7cc55a13-d193-4bcb-8f5e-3e65985c73dd.png)
1) **data_preparation.ipynb** prepares the data. It reads in a json file, coverts it into csv file, splits it into test and train set and writes back the train_qa.csv, and test_qa.csv. The files available in _/contents_ directory.
2) **document_encoder.ipynb** encodes the documents, using BERT model. This BERT encoder is never trained and only pre-trained embeddings were taken for the documents.
3) **question_encoder.ipynb** encodes the queries. This BERT encoder is trained using the corresponding context of the query to calculate the loss function. The loss function is -ve log of the dot product ot the query embedding with the document embedding. The encoder tries to minimize that.
4) **DPR_module.ipynb** This module works to retrieve the top k documents relevant to the query once the query encoder is satisfactorily trained. FAISS is used here to retrieve the relevant documents. The program then goes on to create a new jason file which has the query and top k documents as the input_text and corresponding answer as the out_text.
5) **gen_BART.ipynb** This module reads this new dataset and tries to train the answer generator using BART.
Once generator module is trained, it is fed the questions as input and answers are predicted. These predicted answers are then compared with actual answers and bertscore is calculated.

## Key Implementation Details to be noted
1) For both question and document encoder, *BertModel* and *BertTokenizer* were used with *bert-base-uncased*. \[CLS\] token from the embedding layer is picked up.</br> 
Document encoder is never trained. So the output of the model is taken and \[CLS\] token of dimension 768 is extracted. A tensor of size \[no_of_training_examples, 768\] of all documents encodings is created. Since CUDA was running out of memory, this was done in two parts. So encoded1_z.pt, encoded2_z.pt were created train dataset. encoded3_z.pt is encodings for test set.</br> 
2) For Question Encoder:
- A TensorDataset of encodings of documents were created to be able to load encoded documents (created in document encoder) batch-wise. 
- A SequentialSampler is used to load both question samples and document encoding samples to maintain the order.
- \[CLS\] token of dimension 768 is taken from the output of question encoder model. This is a tensor of size \[batch_size, 768\]. 
- Negative Log of dot product of question and document encoding is the loss, which is back propagated to train the model. The loss reported in the notbook is total loss for 14011 samples. The average loss can be obtained from there by dividing with the size of training set. The averge training loss is reported below for some epochs

epoch | Training loss | Test loss
------|-----------|------
1  | 0.2588	| 
13 | 0.129994 | 0.3999
23 | 0.095982 | 0.4203

- Optimizer used is AdamW with learning rate 5e-5.
- The model was trained for 23 (in steps of 3 + 10 + another 10) epochs. This is not a magic number but a careful training so that CUDA doesn't run of memory. 
- The training loss steadily decreased with few bumps on the way as can be seen in the image below:

![image](https://user-images.githubusercontent.com/82941475/133927565-809fffa8-a5d6-401b-bce2-068df3ab41d3.png)

- After every 10 epochs, the model was evaluated on test set, to evaluate overfitting.
- This trained model was then passed to DPR_module to be retrieve top k relevant documents (k was kept to be 3, with no particular reason but a starting number to be played around with, but later on taken as it is.) </br>
3) DPR Module
- Here we take already trained question encoder, and document embeddings.
- Use FAISS as the similarity module as this is more optimized than calculating the cosine similarity of each document with a given query.
- The document embeddings, returned by document encoder are loaded in a tensor as an index needs to be created. There are varieties of indexes available. I have chosen IndexFlatL1 to start with. This is another aspect which can be played around with, which was the idea initially, but later on this was kept as it is.
- The FAISS index expects a numpy array. So document encodings were converted into numpy array before building the index.
- The query encodings from the trained question enocder were taken, converted into numpy array and passed to the index to return the top k documents similar to query.
- The index returns the distance and document indexes. These indexes were taken and the corresponding documents were picked up from documents' dataset.
- A new json file with query+document(1)+document(2)+document(3) as *in_text* and answer to the query as *out_text* was created to be passed on BART generator.
- This way separately train set (newqa_train.json) and test set (newqa_test.json) were created. These files are available in _/content_ directory.

4) BART Generator
- *BartTokenizer* and *BartForConditionalGeneration* with 'facebook/bart-large' were used.
- This new dataset created in DPR Module is read, tokenized and a dataset for both training (train_dataset) and evaluation (test_dataset) were created. 
- The answer to the query was taken as *labels* to pass on to decoder of BART to calculate the loss. Important point to note here is to makes sure that the decoder inputs are correctly shifted and still include the padding tokens while in the *labels* the padding tokens are replaced by -100 so that they are ignored in the model loss. *shift_tokens_right* in gen_BART.ipynb takes care of this aspect.
- The model was trained and the loss steadily goes down as can be seen below :
![image](https://user-images.githubusercontent.com/82941475/133927816-af1558c5-9054-4bcb-97a0-2901347ac24f.png)
- Overall training loss reported was **3.02092201767093**. The other related metrics reported are in the notbook gen_BART.ipynb. The model reported a test loss of **2.4269748294**, confirming that the model is not overfitting to the training data.

This model was then used for generating answers for the given questions. The predictions were done for test set, and bert score was calculated. </br>
The bertscore reported is (hug_trans=4.10.2): **Precision = 0.825757,  Recall = 0.808534 and F1 Score = 0.816605100.** 

A review of predictions (_/content/predictions.txt_) revealed that predictions could be better.

## Further Work and possible improvements

The model can be further improved by working on possibly the following aspects: </br>
_There may be other aspects and features and solutions which can be explored and investigated._</br>
a) a better question encoder. Cleary if the question encoder was better trained (for 100 or more epochs), question encodings could have been better leading to better match of the documents retrieved from FAISS.</br>
b) BART generator. If resources and could have permitted, the model could be trained for 2 or 3 epochs. (each epoch takes arounf 3 hours and more than the disk space available in colab pro. Some of the check-points were actually deleted while the training was going on to create space on the disk.)</br>
c) Both the question encoder, BART generator together can be trained better for this overall work to be put to some use.</br>
d) A better dataset</br>

### Key Takeaway:
Complete framework is ready and flow explained above in the image. The framework can be used on a different dataset and evaluated and also can be further trained.

### Key Learnings
Lots of hits and trials and head scratching happened in the beginning with things slowly settling down and fog being lifted.
1) First and most important, to be able to research and play around in the NLP arena, resources must be backed up by a large corporation with dedicated funds. At the individual level, with even colab pro, it was not an enough. 
2) Various concepts were cleared and how to read the model details and to be able to utilize the relevant return field values was one major accomplishment for me.

### Acknowledgements
Heartfelt gratitude to Rohan Shravan, for providing this opportunity, for guidance and teaching this wonderful course. Thank you. </br>
I would to give tons of thanks to my teammates, Chaitanya, Pralay and Pallavi for guiding me throughout, coming to my rescuse whenever I shouted out on the group for help. A big-big thank you to all of you, without which I would not have managed to come this far. </br>
I am gratetful to my pytorch documentation team for some wonderful discussions and ideas.</br> 
Special thanks to  Santosh from END2.0 group and Pralay Ramteke from my team for sharing some insights, wonderful discussions and keeping the momentum for the project just when I would almost give up. Thanks! </br>
Last but not least, many thanks to whole of END2.0 team, who helped in one way or the other for me to be able to reach here. Thank you!


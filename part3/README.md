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

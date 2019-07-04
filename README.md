# Kaggle Competitions Toxic Classification


## Overview
In this competition, we need to detect toxic comments, we use NLP technique and deep learning to build classification model

## Installation
```
pip install -r requirement  
```



## Preprocessing
We try to convert all sentecne to lower case in LSTM model, but normal case gets highly score in LB than lower case. 

Preprocessing as follows:
 *  all http(url) were substituted to url
 *  all emoji substitute to ' '
 *  using flashtext to find mispell words and replace to true words
 *  all emoji were substituted by ' '
 *  \n\t were substituted by ' '
 *  \s{2,} were substituted by ' '

## Get statistics features
We try to get some statistics feature and put in LSTM model training

We get statistics feature as follows:
* swear word 
* upper word 
* uniqen word
* emoji
* characters 


## Embedding
We used pretrainned word embedding as follows:
* Fasttext
* Glove

In our works, fasttext is a little bit better than Glove.
We didn't concatenate fasttext and Glove due to time consuming. (However, in the nearly end of the competition, everyone used BERT model haha.)


## Model
### LSTM model
Our lstm model are different with public version, it comsisted of lstm cells without gru cells.
* Attention didn't improve LB significantly.
* Spatial Dropout had improvement in LB.
* blending of three models, each lstm model got the LB 0.935x~0.938x. After blended three models, we got the LB 0.93963.


### BERT model

We used pretrained bert model from: [pytorch-pretrained-BERT](https://github.com/huggingface/pytorch-pretrained-BERT) and `BertForSequenceClassification` for sequence classification.
* The result with text preprocessing or without text preprocessing are similar.

* The batch size from 16 to 32 improve the LB. I think the batch size significantly influnces acurracy
* The learning rate we set is `2e-5`
* We got the single model with LB 0.9415x~0.94220
* We ensemble five single BERT model and got LB 0.94294


### GPT2 model
* We only got LB around 0.938 in single GPT2. Therefore we focused on training BERT models. 

### Ensemble model

* We used ensemble of 3 LSTM models and ensemble of 5 BERT models and blended them with weights 0.3 and 0.7 respectively.
* In the end, we got LB 0.9443


## References 
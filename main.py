# -*- coding: utf-8 -*
import re
import os
import random

import torch
import unidecode
import numpy as np 
import pandas as pd 
from flashtext import KeywordProcessor

from models.lstm import NeuralNet
from models.bert import bertClassifier
from preprocessing.preprocessing import content_preprocessing


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

### use in main.py
if __name__ == '__main__':

  seed = 9635
  seed_everything(seed)
  print('preprocessing')
  MISSPELL_PATH = open('./preprocessing/mispell.txt', 'r')
  mispell_dict = dict()
  for misspell_word in MISSPELL_PATH:
    words = misspell_word.split(',')
    mispell_dict[words[0]] = words[1].strip(' |\n')


  mix_mispell_dict = {}
  for k, v in mispell_dict.items():
      mix_mispell_dict[k] = v
      mix_mispell_dict[k.lower()] = v.lower()
      mix_mispell_dict[k.upper()] = v.upper()
      mix_mispell_dict[k.capitalize()] = v.capitalize()

  kp = KeywordProcessor(case_sensitive=True)
  for k, v in mix_mispell_dict.items():
      kp.add_keyword(k, v) 


  sentences = ['motherfuckeraif this is a test ! ', 'i am a robot', 'i want to silver', 'this is a test, trumpdon !, trumpland, sallary']
  sentences = pd.DataFrame(sentences, columns=['comment_text'])
  print('this is a test case sentence !')
  print('clean text')
  sentences = sentences['comment_text'].apply(lambda sentence : content_preprocessing(sentence, kp))

  print(sentences)
  print('---'*30)

  print('lstm')
  EPOCHS = 5
  BATCH_SIZE = 1024
  MAX_LEN = 220
  
  embedding_matrix = np.random.random([1000,300])
  net = NeuralNet(embedding_matrix, 128, 220)
  print(net)
  
  """

  train lstm model

  """
  print('---'*30)


  print('bert')
  EPOCHS = 1
  BATCH_SIZE = 32
  MAX_LEN = 220
  accumulation_steps = 1
  bertClassifier = bertClassifier('bert-base-uncased', 1, 'BertAdam', MAX_LEN)

  print('sentences convert')
  x_train = bertClassifier.convert_lines(list(sentences))
  print(x_train)
  print('---'*30)

  print('load bert network architecture and get init weights from pretrained model')
  net = bertClassifier.buildBertNet()
  print(net)
  print('---'*30)

  print('get bert optimizer')
  optimizer =  bertClassifier.buildOptimizer(net, EPOCHS, BATCH_SIZE, len(x_train), accumulation_steps)
  print(optimizer)
  print('---'*30)

  """
  
  fine tune bert model

  """




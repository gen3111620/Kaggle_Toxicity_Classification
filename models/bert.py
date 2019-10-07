# -*- coding: utf-8 -*
import numpy as np
from pytorch_pretrained_bert import BertTokenizer, BertForSequenceClassification, BertAdam, OpenAIAdam



class bertClassifier():
  def __init__(self, model_cased, num_labels, optimizer, max_seq_length ):
      """
      The class can do bert tokenzier and build bert optimizer 
      input param :
      1. model_cased : 'bert-base-uncased' or 'bert-base-cased'
      2. num_labels : numbers of labels 
      3. optimizer : BertAdam or OpenAIAdam
      4. setences : a list of sentence
      5. max_sql_length : define tokenizer max length

      """
      assert model_cased == 'bert-base-uncased' or model_cased == 'bert-base-cased' , \
        'only accept bert-base-uncased or bert-base-cased'
      self.model_cased = model_cased

      assert str(num_labels).isdigit(), 'num_labels error, only accept number'
      self.num_labels = num_labels

      assert optimizer == 'BertAdam' or optimizer == 'OpenAIAdam', \
       'only accept BertAdam or OpenAIAdam'
      self.optimizer = optimizer

  #     assert isinstance(sentences, (list, tuple)), 'sentences error, only accept list or tuple'
  #     self.sentences = sentences

      assert isinstance(max_seq_length, int), 'max_seq_length error, only accept number'
      self.max_seq_length = max_seq_length

  def buildBertNet(self):
      return BertForSequenceClassification.from_pretrained(self.model_cased, num_labels=self.num_labels)


  def convert_lines(self, sentences):

      """
      bert convert token to ids
      input : 

      1. self.sentences : a list of sentence
      2. self.max_seq_length : is like keras tokenizer padding, 
      3. tokenizer : bert tokenizer, 'bert-base-uncased' or 'bert-base-cased'

      output : all sentence(words) convert to ids
      """ 
      tokenizer = BertTokenizer.from_pretrained(self.model_cased)

      self.max_seq_length -= 2
      all_tokens = []
      longer = 0
      for text in sentences:
          tokens_a = tokenizer.tokenize(text)
          if len(tokens_a) > self.max_seq_length:
              tokens_a = tokens_a[:self.max_seq_length]
              longer += 1
          one_token = tokenizer.convert_tokens_to_ids(["[CLS]"]+tokens_a+["[SEP]"])+[0] * (self.max_seq_length - len(tokens_a))
          all_tokens.append(one_token)

      return np.array(all_tokens)

  def buildOptimizer(self, neural, epochs, batch_size, train_total, accumulation_steps, lr=1e-5, warmup=0.1):

      """
      build bert optimizer

      """
      no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

      param_optimizer = list(neural.named_parameters())

      optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
      ]

      num_train_optimization_steps = int(epochs*train_total/batch_size/accumulation_steps)

      if self.optimizer == 'BertAdam':
          return BertAdam(optimizer_grouped_parameters, lr=lr, warmup=warmup, t_total=num_train_optimization_steps)
      else:
          return OpenAIAdam(optimizer_grouped_parameters, lr=lr, warmup=warmup, t_total=num_train_optimization_steps)

  """

  bert optimizer and tokenizer done.
  if you have bert pretrained model, you can load your model and weights
  ex : net.load_state_dict(torch.load("../input/bert-cased/bert_pytorch.pt"))
  
  then
  1. net.cuda()
  2, define loss fouction
  3. do bert fine tune, like lstm training normally
  4. save your model
  

  """

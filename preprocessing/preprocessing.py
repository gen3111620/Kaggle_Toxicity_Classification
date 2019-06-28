# -*- coding: utf-8 -*
import re
import os
import random
import unidecode

import numpy as np 
import pandas as pd 
from flashtext import KeywordProcessor
from statistics_features import statistics_upper_words
from statistics_features import statistics_unique_words
from statistics_features import statistics_swear_words
from statistics_features import statistics_characters_nums

EMOJI_RE = re.compile(u'['u'\U00010000-\U0010ffff' 
                          u'\U0001F600-\U0001F64F'
                          u'\U0001F300-\U0001F5FF'
                          u'\U0001F30D-\U0001F567'
                          u'\U0001F680-\U0001F6FF'
                          u'\u2122-\u2B55]', re.UNICODE
                      )

PUNCTS = {
            '》', '〞', '¢', '‹', '╦', '║', '♪', 'Ø', '╩', '\\', '★', '＋', 'ï', '<', '?', '％', '+', '„', 'α', '*', '〰', '｟', '¹', '●', '〗', ']', '▾', '■', '〙', '↓', '´', '【', 'ᴵ',
            '"', '）', '｀', '│', '¤', '²', '‡', '¿', '–', '」', '╔', '〾', '%', '¾', '←', '〔', '＿', '’', '-', ':', '‧', '｛', 'β', '（', '─', 'à', 'â', '､', '•', '；', '☆', '／', 'π',
            'é', '╗', '＾', '▪', ',', '►', '/', '〚', '¶', '♦', '™', '}', '″', '＂', '『', '▬', '±', '«', '“', '÷', '×', '^', '!', '╣', '▲', '・', '░', '′', '〝', '‛', '√', ';', '】', '▼',
            '.', '~', '`', '。', 'ə', '］', '，', '{', '～', '！', '†', '‘', '﹏', '═', '｣', '〕', '〜', '＼', '▒', '＄', '♥', '〛', '≤', '∞', '_', '[', '＆', '→', '»', '－', '＝', '§', '⋅', 
            '▓', '&', 'Â', '＞', '〃', '|', '¦', '—', '╚', '〖', '―', '¸', '³', '®', '｠', '¨', '‟', '＊', '£', '#', 'Ã', "'", '▀', '·', '？', '、', '█', '”', '＃', '⊕', '=', '〟', '½', '』',
            '［', '$', ')', 'θ', '@', '›', '＠', '｝', '¬', '…', '¼', '：', '¥', '❤', '€', '−', '＜', '(', '〘', '▄', '＇', '>', '₤', '₹', '∅', 'è', '〿', '「', '©', '｢', '∙', '°', '｜', '¡', 
            '↑', 'º', '¯', '♫'
          }

def clean_punct(text):
  text = str(text)
  for punct in PUNCTS:
    text = text.replace(punct, ' {} '.format(punct))
  
  return text


def content_preprocessing(text, kp, statistics_features=False):
  """

  clean text step by step:

  1. all http(url) were substituted to url
  2. all emoji substitute to ' '
  3. using flashtext to find mispell words and replace to true words
  4. all emoji were substituted by ' '
  5. /n/t were substituted by ' '
  6. /s{2,} were substituted by ' ' 
  
  """

  ### if statistics_features is True, will statstics the sentence features
  if statistics_features is True:
    emoji_num = len(EMOJI_RE.findall(text))
    swear_words_num = statistics_swear_words(text, SWEAR_WORDS)
    upper_count = statistics_upper_words(text)
    characters_num = statistics_characters_nums(text)
    unique_words_num = statistics_unique_words(text)
    

  text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', 'url', text)
  text = kp.replace_keywords(text)
  text = clean_punct(text)
  text = EMOJI_RE.sub(' ', text)
  text = re.sub(r'\n\r', ' ', text)
  text = re.sub(r'\s{2,}', ' ', text)

  if statistics_features:
    return text, len(text.split()), emoji_num, swear_words_num, upper_count, characters_num, unique_words_num

  return text
                



### test code
if __name__ == '__main__':
  MISSPELL_PATH = open('mispell.txt', 'r')

  mispell_dict = dict()
  for misspell_word in MISSPELL_PATH:
    words = misspell_word.split(',')
    mispell_dict[words[0]] = words[1].strip(' |\n')

  SWEAR_WORDS_PATH = './swear_words.txt'

  SWEAR_WORDS = []
  with open(SWEAR_WORDS_PATH, 'r') as f:
      for token in f:
          SWEAR_WORDS.append(re.sub('\n', '', token))

  mix_mispell_dict = {}
  for k, v in mispell_dict.items():
      mix_mispell_dict[k] = v
      mix_mispell_dict[k.lower()] = v.lower()
      mix_mispell_dict[k.upper()] = v.upper()
      mix_mispell_dict[k.capitalize()] = v.capitalize()

  kp = KeywordProcessor(case_sensitive=True)
  for k, v in mix_mispell_dict.items():
      kp.add_keyword(k, v) 

  # kp = getMisspellWords(MISSPELL_PATH)

  print('this is a test case sentence !')
  print('clean text')
  print(content_preprocessing('this is a test, trumpdon !, trumpland, sallary', kp))
  print('---'*30)
  print('emoji')
  print('😁 👸 🤴 👎 👊 ✊')
  print('emoji replace')
  print(EMOJI_RE.sub(' ', '😁 👸 🤴 👎 👊 ✊'))
  print('---'*30)
  print('test get statstics features')
  print('This is a test, trumpdon !, trumpland, sallary')
  print(content_preprocessing('This is a test, trumpdon !, trumpland, sallary 😁 👸 🤴 👎 ', kp, statistics_features=True))


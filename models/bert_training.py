#!/usr/bin/env python
#coding=utf-8

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os

# Any results you write to the current directory are saved as output.

import random
import unidecode
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import time
import torch.nn.functional as F
from keras.preprocessing import text, sequence
from torch.utils.data import Dataset, DataLoader,TensorDataset
from sklearn.model_selection import train_test_split
import re
from gensim.models import KeyedVectors
from flashtext import KeywordProcessor


os.system('pip install pytorch-pretrained-bert')
from pytorch_pretrained_bert import BertTokenizer, BertForSequenceClassification, BertAdam


'''

bert v10
0.5 epoch

'''



BATCH_SIZE = 32
EPOCHS = 1
MAX_LEN = 220
NUM_MODEL = 1
accumulation_steps = 1
lr = 2e-5
seed = 1111

def seed_everything(seed=seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
seed_everything()

def get_coefs(word, *arr):
    return word, np.asarray(arr, dtype='float32')

def load_embeddings(path):
    with open(path, encoding="utf8", errors='ignore') as f:
        return dict(get_coefs(*line.strip().split(' ')) for line in f)


# Converting the lines to BERT format
# Thanks to https://www.kaggle.com/httpwwwfszyc/bert-in-keras-taming
def convert_lines(example, max_seq_length,tokenizer):
    max_seq_length -=2
    all_tokens = []
    longer = 0
    for text in example:
        tokens_a = tokenizer.tokenize(text)
        if len(tokens_a)>max_seq_length:
            tokens_a = tokens_a[:max_seq_length]
            longer += 1
        one_token = tokenizer.convert_tokens_to_ids(["[CLS]"]+tokens_a+["[SEP]"])+[0] * (max_seq_length - len(tokens_a))
        all_tokens.append(one_token)
    return np.array(all_tokens)


def custom_loss(data, targets):

	''' Define custom loss function for weighted BCE on 'target' column '''

	bce_loss = nn.BCEWithLogitsLoss(weight=targets[:,1])(data[:,0],targets[:,0])
	return bce_loss
    

# swear_words = []
# with open(SWEAR_WORDS_PATH, 'r') as f:
#     for token in f:
#         swear_words.append(re.sub('\n', '', token))
# swear_words.extend(['<q>', '<a>', '<s>', '<x>', '<c>', '<b>','<n>', 'trump'])


punc_sign = r"\ə\ᴵ\'∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–&\\…\/\{\}\''\[\]\_\/\@\$\%\^\&\*\(\)\+\#\:\!\-\;\!\"\\(\),\.?'+`~$=|•！？。＂＃＄％＆＇（）＊＋，－／：；<>＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟‧﹏"

puncts = [',', '.', '"', ':', ')', '(', '!', '?', '|', ';', "'", '$', '&',
    '/', '[', ']', '>', '%', '=', '#', '+', '\\', '•',  '~', '@', '£',
    '·', '_', '{', '}', '©', '^', '®', '`',  '<', '→', '°', '€', '™', '›',
    '♥', '←', '×', '§', '″', '′', 'Â', '█', '½', 'à', '…', '“', '★', '”',
    '–', '●', 'â', '►', '−', '¢', '²', '¬', '░', '¶', '↑', '±', '¿', '▾',
    '═', '¦', '║', '―', '¥', '▓', '—', '‹', '─', '▒', '：', '¼', '⊕', '▼',
    '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲',
    'è', '¸', '¾', 'Ã', '⋅', '‘', '∞', '∙', '）', '↓', '、', '│', '（', '»',
    '，', '♪', '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø',
    '¹', '≤', '‡', '√', '«', '»', '´', 'º', '¾', '¡', '§', '£', '₤', 'ə', '√',
    'ᴵ', '∞', 'θ', '÷', 'α', '•', 'à', '−', 'β', '∅', '³', 'π', '‘', '₹', '´', '£', '€',
    '×','™', '√', '²', '—', '…', ':', ';', '•', '！', '?', '$', '＄', '％', '＆', '（', '）', '-', '*']

def clean_text(x):
    x = str(x)
    for punct in puncts:
    	x = x.replace(punct, f' {punct} ')
    return x


mispell_dict = {'colour': 'color', 'centre': 'center', 'favourite': 'favorite', 'travelling': 'traveling', 
                'counselling': 'counseling', 'theatre': 'theater', 'cancelled': 'canceled', 'labour': 'labor', 
                'organisation': 'organization', 'wwii': 'world war 2', 'citicise': 'criticize', 'youtu ': 'youtube ', 
                'qoura': 'quora', 'sallary': 'salary', 'Whta': 'What', 'narcisist': 'narcissist', 'howdo': 'how do', 
                'whatare': 'what are', 'howcan': 'how can', 'howmuch': 'how much', 'howmany': 'how many', 'em': 'them',
                'whydo': 'why do', 'doI': 'do I', 'theBest': 'the best', 'howdoes': 'how does', 'mastrubation': 'masturbation', 
                'mastrubate': 'masturbate', "mastrubating": 'masturbating', 'pennis': 'penis', 'etherium': 'ethereum', 
                'narcissit': 'narcissist', 'bigdata': 'big data', '2k17': '2017', '2k18': '2018', '2k19':'2019', 'qouta': 'quota', 
                'exboyfriend': 'ex boyfriend', 'airhostess': 'air hostess', "whst": 'what', 'watsapp': 'whatsapp', 
                'demonitisation': 'demonetization', 'demonitization': 'demonetization', 
                'demonetisation': 'demonetization', 'pokémon': 'pokemon', 'n*gga':'nigga', 'p*':'pussy', 
                'b***h':'bitch', 'a***h****':'asshole', 'a****le-ish':'asshole', 'b*ll-s***':'bullshit', 'd*g':'dog', 
                'st*up*id':'stupid','d***':'dick','di**':'dick',"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have",
                "couldn't": "could not", "didn't": "did not",  "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", 
                "haven't": "have not", "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", 
                "how's": "how is",  "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would", 
                "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would", "it'd've": "it would have", 
                "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have","mightn't": "might not",
                "mightn't've": "might not have", "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock",
                "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have",
                "she'll": "she will", "she'll've": "she will have", "she's": "she is", "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as",
                "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would", "there'd've": "there would have", "there's": "there is", "here's": "here is",
                "they'd": "they would", "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not",
                "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have",
                "what're": "what are",  "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is", "where've": "where have", "who'll": "who will",
                "who'll've": "who will have", "who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have", "would've": "would have",
                "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have","you'd": "you would",
                "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have", "you're": "you are", "you've": "you have", 'colour': 'color', 'centre': 'center', 'favourite': 'favorite', 'travelling': 'traveling',
                'counselling': 'counseling', 'theatre': 'theater', 'cancelled': 'canceled', 'labour': 'labor', 'organisation': 'organization', 'wwii': 'world war 2', 'citicise': 'criticize', 'youtu ': 'youtube ', 'Qoura': 'Quora',
                'sallary': 'salary', 'Whta': 'What', 'narcisist': 'narcissist', 'howdo': 'how do', 'whatare': 'what are', 'howcan': 'how can', 'howmuch': 'how much', 'howmany': 'how many', 'whydo': 'why do', 'doI': 'do I', 'theBest': 'the best',
                'howdoes': 'how does', 'mastrubation': 'masturbation', 'mastrubate': 'masturbate', "mastrubating": 'masturbating', 'pennis': 'penis', 'Etherium': 'Ethereum', 'narcissit': 'narcissist', 'bigdata': 'big data',
                '2k17': '2017', '2k18': '2018', 'qouta': 'quota', 'exboyfriend': 'ex boyfriend', 'airhostess': 'air hostess', "whst": 'what', 'watsapp': 'whatsapp', 'demonitisation': 'demonetization',
                'demonitization': 'demonetization', 'demonetisation': 'demonetization','\u200b': ' ', '\ufeff': '', 'करना': '', 'है': '',
                'sh*tty': 'shitty','s**t':'shit',
                'nigg*r':'nigger','bulls**t':'bullshit','n*****':'nigger',
                'p*ssy':'pussy','p***y':'pussy',
                'f***':'fuck','f*^k':'fuck','f*cked':'fucked','f*ck':'fuck','f***ing':'fucking','F*CKING': 'fucking',
                'sh*t':'shit', 'su*k':'suck', 'a**holes':'assholes','a**hole':'asshole',
                'di*k':'dick', 'd*ck': 'dick', 'd**k':'dick', 'd***':'dick',
                'bull**it':'bullshit', 'c**t':'cunt', 'cu*t':'cunt', 'c*nt':'cunt','troʊl':'trool',
                'trumpian':'bombast','realdonaldtrump':'trump','drumpf':'trump','trumpist':'trump',
                "i'ma": "i am","is'nt": "is not","‘I":'I',
                'ᴀɴᴅ':'and','ᴛʜᴇ':'the','ʜᴏᴍᴇ':'home','ᴜᴘ':'up','ʙʏ':'by','ᴀᴛ':'at','…and':'and','civilbeat':'civil beat',\
                'TrumpCare':'Trump care','Trumpcare':'Trump care', 'OBAMAcare':'Obama care','ᴄʜᴇᴄᴋ':'check','ғᴏʀ':'for','ᴛʜɪs':'this','ᴄᴏᴍᴘᴜᴛᴇʀ':'computer',\
                'ᴍᴏɴᴛʜ':'month','ᴡᴏʀᴋɪɴɢ':'working','ᴊᴏʙ':'job','ғʀᴏᴍ':'from','Sᴛᴀʀᴛ':'start','gubmit':'submit','CO₂':'carbon dioxide','ғɪʀsᴛ':'first',\
                'ᴇɴᴅ':'end','ᴄᴀɴ':'can','ʜᴀᴠᴇ':'have','ᴛᴏ':'to','ʟɪɴᴋ':'link','ᴏғ':'of','ʜᴏᴜʀʟʏ':'hourly','ᴡᴇᴇᴋ':'week','ᴇɴᴅ':'end','ᴇxᴛʀᴀ':'extra',\
                'Gʀᴇᴀᴛ':'great','sᴛᴜᴅᴇɴᴛs':'student','sᴛᴀʏ':'stay','ᴍᴏᴍs':'mother','ᴏʀ':'or','ᴀɴʏᴏɴᴇ':'anyone','ɴᴇᴇᴅɪɴɢ':'needing','ᴀɴ':'an','ɪɴᴄᴏᴍᴇ':'income',\
                'ʀᴇʟɪᴀʙʟᴇ':'reliable','ғɪʀsᴛ':'first','ʏᴏᴜʀ':'your','sɪɢɴɪɴɢ':'signing','ʙᴏᴛᴛᴏᴍ':'bottom','ғᴏʟʟᴏᴡɪɴɢ':'following','Mᴀᴋᴇ':'make',\
                'ᴄᴏɴɴᴇᴄᴛɪᴏɴ':'connection','ɪɴᴛᴇʀɴᴇᴛ':'internet','financialpost':'financial post', 'ʜaᴠᴇ':' have ', 'ᴄaɴ':' can ', 'Maᴋᴇ':' make ', 'ʀᴇʟɪaʙʟᴇ':' reliable ', 'ɴᴇᴇᴅ':' need ',
                'ᴏɴʟʏ':' only ', 'ᴇxᴛʀa':' extra ', 'aɴ':' an ', 'aɴʏᴏɴᴇ':' anyone ', 'sᴛaʏ':' stay ', 'Sᴛaʀᴛ':' start', 'SHOPO':'shop','ᴀ':'A',
                'theguardian':'the guardian','deplorables':'deplorable', 'theglobeandmail':'the globe and mail', 'justiciaries': 'justiciary','creditdation': 'Accreditation',
                'doctrne':'doctrine','fentayal': 'fentanyl','designation-': 'designation','CONartist' : 'con-artist','Mutilitated' : 'Mutilated','Obumblers': 'bumblers',
                'negotiatiations': 'negotiations','dood-': 'dood','irakis' : 'iraki','cooerate': 'cooperate','COx':'cox','racistcomments':'racist comments','envirnmetalists': 'environmentalists',
                'SB91':'senate bill','tRump':'trump','utmterm':'utm term','FakeNews':'fake news','Gʀᴇat':'great','ʙᴏᴛtoᴍ':'bottom','washingtontimes':'washington times','garycrum':'gary crum','htmlutmterm':'html utm term',
                'RangerMC':'car','TFWs':'tuition fee waiver','SJWs':'social justice warrior','Koncerned':'concerned','Vinis':'vinys','Yᴏᴜ':'you',
                'trumpists': 'trump', 'trumpkins': 'trump','trumpism': 'trump','trumpsters':'trump','thedonald':'trump',
                'trumpty': 'trump', 'trumpettes': 'trump','trumpland': 'trump','trumpies':'trump','trumpo':'trump',
                'drump': 'trump', 'dtrumpview': 'trump','drumph': 'trump','trumpanzee':'trump','trumpite':'trump',
                'chumpsters': 'trump', 'trumptanic': 'trump', 'itʻs': 'it is', 'donʻt': 'do not','pussyhats':'pussy hats',
                'trumpdon': 'trump', 'trumpisms': 'trump','trumperatti':'trump', 'legalizefreedom': 'legalize freedom',
                'trumpish': 'trump', 'ur': 'you are','twitler':'twitter','trumplethinskin':'trump','trumpnuts':'trump','trumpanzees':'trump',
                'justmaybe':'just maybe','trumpie':'trump','trumpistan':'trump','trumphobic':'trump','piano2':'piano','trumplandia':'trump',
                'globalresearch':'global research','trumptydumpty':'trump','frank1':'frank','trumpski':'trump','trumptards':'trump',
                'alwaysthere':'always there','clickbait':'click bait','antifas':'antifa','dtrump':'trump','trumpflakes':'trump flakes',
                'trumputin':'trump putin','fakesarge':'fake sarge','civilbot':'civil bot','tumpkin':'trump','trumpians':'trump',
                'drumpfs':'trump','dtrumpo':'trump','trumpistas':'trump','trumpity':'trump','trump nut':'trump','tumpkin':'trump',
                'russiagate':'russia gate','trumpsucker':'trump sucker','trumpbart':'trump bart', 'trumplicrat':'trump','dtrump0':'trump',
                'tfixstupid':'stupid','brexit':'British exit','Brexit':'British exit','trumpelthinskin':'trump', 'americanophobia': 'Anti-Americanism', 
                'magaphants':'anti-trump', 'MAGAphants':'anti-trump', 'klastri':'<b>','cheetolini':'trump','daesh':'ISIS'
               }
               

# mispell_dict2 = {'americanophobia': '<q>', 'klastri':'<s>','thisisurl':'<url>','magaphants':'<x>','cheetolini':'<c>','daesh':'<b>',
#                 'trumpelthinskin':'<n>'}
emoji_re = re.compile(u'['
                        u'\U00010000-\U0010ffff' 
                        u'\U0001F600-\U0001F64F'
                        u'\U0001F300-\U0001F5FF'
                        u'\U0001F30D-\U0001F567'
                        u'\U0001F680-\U0001F6FF'
                        u'\u2122-\u2B55]', re.UNICODE)

kp = KeywordProcessor(case_sensitive=True)
                
mix_mispell_dict = {}
for k, v in mispell_dict.items():
    mix_mispell_dict[k] = v
    mix_mispell_dict[k.lower()] = v.lower()
    mix_mispell_dict[k.upper()] = v.upper()
    mix_mispell_dict[k.capitalize()] = v.capitalize()

for k, v in mix_mispell_dict.items():
    kp.add_keyword(k, v)    
    

# kp2 = KeywordProcessor(case_sensitive=True)
# for k, v in mispell_dict2.items():
#     kp2.add_keyword(k, v)
    

def content_preprocessing(text):
    
    
    # text = text.lower()
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', ' url ', text)
    text = kp.replace_keywords(text)
    # text = re.sub("[%s]+" %punc_sign , ' ' ,text)
    text = clean_text(text)
    # emoji_num = len(emoji_re.findall(text))
    text = emoji_re.sub(' ', text)
    # text = kp2.replace_keywords(text)
    text = re.sub(r'\n\r', '', text)
    text = re.sub(r'\s{2,}', ' ', text)

    return text
	
		
train_df = pd.read_csv("../input/jigsaw-unintended-bias-in-toxicity-classification/train.csv")[900000:]
test_df = pd.read_csv("../input/jigsaw-unintended-bias-in-toxicity-classification/test.csv")

train_df['comment_text'] = train_df['comment_text'].fillna('_##_').values
test_df['comment_text'] = test_df['comment_text'].fillna('_##_').values


# shuffling the data
train_df = train_df.sample(frac=1).reset_index(drop=True)

print("Train shape : ", train_df.shape)
print("Test shape : ", test_df.shape)

### preprocessing
# x_train = train_df["comment_text"]
# x_test = test_df["comment_text"]
x_train = train_df["comment_text"].apply(lambda x: content_preprocessing(x))
y_aux_train = np.array(train_df[['severe_toxicity', 'obscene', 'identity_attack', 'insult', 'threat']])
x_test = test_df["comment_text"].apply(lambda x: content_preprocessing(x))


del test_df

identity_columns = [
    'male', 'female', 'homosexual_gay_or_lesbian', 'christian', 'jewish',
    'muslim', 'black', 'white', 'psychiatric_or_mental_illness']

# Overall
weights = np.ones((len(x_train),)) / 4
# Subgroup
weights += (train_df[identity_columns].fillna(0).values>=0.5).sum(axis=1).astype(bool).astype(np.int) / 4
# Background Positive, Subgroup Negative
weights += (( (train_df['target'].values>=0.5).astype(bool).astype(np.int) +
   (train_df[identity_columns].fillna(0).values<0.5).sum(axis=1).astype(bool).astype(np.int) ) > 1 ).astype(bool).astype(np.int) / 4
# Background Negative, Subgroup Positive
weights += (( (train_df['target'].values<0.5).astype(bool).astype(np.int) +
   (train_df[identity_columns].fillna(0).values>=0.5).sum(axis=1).astype(bool).astype(np.int) ) > 1 ).astype(bool).astype(np.int) / 4

## loss_weight = 3.209226860170181
loss_weight = 1.0 / weights.mean()

y_train = np.vstack([train_df['target'],weights]).T
y_train = np.concatenate((y_train, y_aux_train), axis=1)

del y_aux_train
del train_df


## Tokenize and padding
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
x_train = convert_lines(x_train,MAX_LEN,tokenizer)
x_test = convert_lines(x_test,MAX_LEN,tokenizer)


# shuffling the data
# np.random.seed()
train_idx = np.random.permutation(len(x_train))
x_train = x_train[train_idx]
y_train = y_train[train_idx]

#### Set model parameters
x_test_cuda = torch.tensor(x_test, dtype=torch.long).cuda()
test_data = torch.utils.data.TensorDataset(x_test_cuda)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

final_test = list()

for index in range(NUM_MODEL):

  seed_everything(seed+index)
  
  x_train_fold = torch.tensor(x_train, dtype=torch.long)
  y_train_fold = torch.tensor(y_train, dtype=torch.float)

  train_data = torch.utils.data.TensorDataset(x_train_fold, y_train_fold)

  train_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)

  print("model: {}".format(index))

  net = BertForSequenceClassification.from_pretrained('bert-base-uncased',num_labels=6)
  ## load pretrain model
#   net.load_state_dict(torch.load("../input/bert-model3/bert_pytorch_v3.pt"))
  net.load_state_dict(torch.load("../input/pytorch-943-bert/bert_pytorch.pt"))
  net.cuda()

  loss_fn = torch.nn.BCEWithLogitsLoss(reduction='mean')

  param_optimizer = list(net.named_parameters())
  no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
  optimizer_grouped_parameters = [
      {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
      {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
      ]
  #len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
  num_train_optimization_steps = int(EPOCHS*len(train_data)/BATCH_SIZE/accumulation_steps)

  optimizer = BertAdam(optimizer_grouped_parameters,
                      lr=lr,
                      warmup=0.05,
                      t_total=num_train_optimization_steps)


  test_checkpoint = list()
  loss_checkpoint = list()

  for epoch in range(EPOCHS):  # loop over the dataset multiple times

    start_time = time.time()

    avg_loss = 0.0

    net.train()
    for i, data in enumerate(train_loader):

      # get the inputs
      inputs, labels = data
      inputs = inputs.cuda()
      labels = labels.cuda()

      label1 = labels[:,:2]
      label2 = labels[:,2:]

      ## forward + backward + optimize
      pred = net(inputs)	
      pred1 = pred[:,:1]
      pred2 = pred[:,1:]

      loss1 = custom_loss(pred1, label1)
      loss2 = loss_fn(pred2,label2)
      loss = loss1*loss_weight+loss2

      # zero the parameter gradients
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      avg_loss += loss.item()

    elapsed_time = time.time() - start_time 

    # print('Epoch {}/{} \t loss={:.4f}\t val_loss={:.4f} \t time={:.2f}s'.format(
    #         epoch+1, EPOCHS, avg_loss/len(train_loader),avg_val_loss/len(val_loader), elapsed_time))
    
    # print('Epoch {}/{} \t loss={:.4f} \t time={:.2f}s'.format(
    # 				epoch+1, EPOCHS, avg_loss/len(train_loader), elapsed_time))

    
    torch.save(net.state_dict(), "bert_pytorch.pt")
    net.eval()
    ## inference
    result = list()
    with torch.no_grad():
      for (x_batch,) in test_loader:
        y_pred = net(x_batch)
        y_pred = torch.sigmoid(y_pred.cpu()).numpy()[:,0]
        result.extend(y_pred)

    test_checkpoint.append(result)
    # loss_checkpoint.append(avg_val_loss)

  final_test.append(test_checkpoint[-1])


## submission
submission = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/sample_submission.csv', index_col='id')
submission['prediction'] = np.mean(final_test, axis=0)
submission.reset_index(drop=False, inplace=True)
submission.to_csv('submission.csv', index=False)
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
import sys
import torch.nn.functional as F
from keras.preprocessing import text, sequence
from torch.utils.data import Dataset, DataLoader,TensorDataset
from sklearn.model_selection import train_test_split
import re
from gensim.models import KeyedVectors
from flashtext import KeywordProcessor


# print(os.listdir("../input/bert-pretrained-models/bert-pretrained-models/bert-pretrained-models/uncased_L-12_H-768_A-12"))

CRAWL_EMBEDDING_PATH = '../input/fasttext-pretrained/crawl-300d-2M.vec'
package_dir = "../input/pytorchpretrainedbert/pytorch-pretrained-bert/pytorch-pretrained-BERT"


sys.path.append(package_dir)
from pytorch_pretrained_bert import BertTokenizer, BertForSequenceClassification, BertAdam, BertConfig


## parameter setting
BATCH_SIZE = 32
MAX_LEN = 220


def get_coefs(word, *arr):
    return word, np.asarray(arr, dtype='float32')

def load_embeddings(path):
    with open(path, encoding="utf8", errors='ignore') as f:
        return dict(get_coefs(*line.strip().split(' ')) for line in f)

def build_matrix(word_index, path):

    """
    https://www.kaggle.com/bminixhofer/simple-lstm-pytorch-version
    """

    embedding_index = load_embeddings(path)
    embedding_matrix = np.zeros((len(word_index) + 1, 300))
#     unknown_vector = np.zeros((300,), dtype=np.float32) - 1.
    
    unknown_words = []

    for word, i in word_index.items():
        
        word = re.sub('[0-9]', '', word)
        if word in embedding_index:
            embedding_matrix[i] = embedding_index[word]
            continue
        if word.upper() in embedding_index:
            embedding_matrix[i] = embedding_index[word.upper()]
            continue
        if word.capitalize() in embedding_index:
            embedding_matrix[i] = embedding_index[word.capitalize()]
            continue
        if unidecode.unidecode(word) in embedding_index:
            embedding_matrix[i] = embedding_index[unidecode.unidecode(word)]
            continue
        
#         embedding_matrix[i] = unknown_vector
        unknown_words.append(word)
            
    return embedding_matrix, unknown_words

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

class Attention(nn.Module):
	def __init__(self, feature_dim, step_dim, bias=True, **kwargs):
		super(Attention, self).__init__(**kwargs)
		
		self.supports_masking = True
		self.bias = bias
		self.feature_dim = feature_dim
		self.step_dim = step_dim
		self.features_dim = 0
		
		weight = torch.zeros(feature_dim, 1)
		nn.init.xavier_uniform_(weight)
		self.weight = nn.Parameter(weight)
		
		if bias:
				self.b = nn.Parameter(torch.zeros(1))
        
	def forward(self, x, mask=None):

		feature_dim = self.feature_dim
		step_dim = self.step_dim

		eij = torch.mm(
				x.contiguous().view(-1, feature_dim), 
				self.weight
		).view(-1, step_dim)
		
		if self.bias:
				eij = eij + self.b
				
		eij = torch.tanh(eij)
		a = torch.exp(eij)
		
		if mask is not None:
				a = a * mask

		a = a / torch.sum(a, 1, keepdim=True) + 1e-10

		weighted_input = x * torch.unsqueeze(a, -1)
		return torch.sum(weighted_input, 1)

class SpatialDropout(nn.Module):

	def __init__(self,p):
		super(SpatialDropout, self).__init__()
		self.dropout = nn.Dropout2d(p)

	def forward(self, x):

			x = x.permute(0, 2, 1)   # convert to [batch, feature, timestep]
			x = self.dropout(x)
			x = x.permute(0, 2, 1)   # back to [batch, timestep, feature]
			return x

class NeuralNet(nn.Module):

	def __init__(self,embedding_matrix,num_unit):
		super(NeuralNet, self).__init__()
		self.max_feature = embedding_matrix.shape[0]
		self.embedding_size = embedding_matrix.shape[1]
		self.embedding = nn.Embedding(self.max_feature, self.embedding_size)
		self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
		self.embedding.weight.requires_grad = False
		self.embedding_dropout = SpatialDropout(0.2)
		self.lstm1 = nn.LSTM(self.embedding_size, num_unit, bidirectional=True, batch_first=True)
		self.lstm2 = nn.LSTM(num_unit*2, int(num_unit/2), bidirectional=True, batch_first=True)
		self.attention = Attention(num_unit, MAX_LEN)
		self.linear1 = nn.Linear(num_unit*3, num_unit)
		self.linear2 = nn.Linear(num_unit*3, num_unit)
		self.linear_out = nn.Linear(num_unit, 1)
		self.linear_aux_out = nn.Linear(num_unit, 5)

	def forward(self, x):

		h_embedding = self.embedding(x)
		h_embedding = self.embedding_dropout(h_embedding)
		h_lstm1, _ = self.lstm1(h_embedding)
		h_lstm2, _ = self.lstm2(h_lstm1)

		# attention
		att = self.attention(h_lstm2)

		# global average pooling
		avg_pool = torch.mean(h_lstm2, 1)

		# global max pooling
		max_pool, _ = torch.max(h_lstm2, 1)

		# concatenation
		h = torch.cat((max_pool, avg_pool, att), 1)

		h_linear1 = F.relu(self.linear1(h))
		h_linear2 = F.relu(self.linear2(h))

		out1 = F.sigmoid(self.linear_out(h_linear1))
		out2 = F.sigmoid(self.linear_aux_out(h_linear2))

		return out1, out2

## read data
test_df = pd.read_csv("../input/jigsaw-unintended-bias-in-toxicity-classification/test.csv")
test_df['comment_text'] = test_df['comment_text'].fillna('_##_').values
print("Test shape : ", test_df.shape)

## preprocessing
x_test = test_df["comment_text"].apply(lambda x: content_preprocessing(x))

## Tokenize and padding
BERT_MODEL_PATH = '../input/bert-pretrained-models/uncased_l-12_h-768_a-12/uncased_L-12_H-768_A-12/'
tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_PATH)
x_test = convert_lines(x_test,MAX_LEN,tokenizer)


x_test_cuda = torch.tensor(x_test, dtype=torch.long).cuda()
test_data = torch.utils.data.TensorDataset(x_test_cuda)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

## load fine-tuned model
bert_config = BertConfig('../input/bert-pretrained-models/uncased_l-12_h-768_a-12/uncased_L-12_H-768_A-12/bert_config.json')
net = BertForSequenceClassification(bert_config,num_labels=6)
net.load_state_dict(torch.load("../input/bert-model3/bert_pytorch_v3.pt"))
net.cuda()

## inference
net.eval()
result_1 = list()
with torch.no_grad():
  for (x_batch,) in test_loader:
    y_pred = net(x_batch)
    y_pred = torch.sigmoid(y_pred.cpu()).numpy()[:,0]
    result_1.extend(y_pred)
result_1 = np.array(result_1)



net = BertForSequenceClassification(bert_config,num_labels=6)
net.load_state_dict(torch.load("../input/bert-model4/bert_pytorch_v4.pt"))
net.cuda()

## inference
net.eval()
result_2 = list()
with torch.no_grad():
  for (x_batch,) in test_loader:
    y_pred = net(x_batch)
    y_pred = torch.sigmoid(y_pred.cpu()).numpy()[:,0]
    result_2.extend(y_pred)
result_2 = np.array(result_2)

# net = BertForSequenceClassification(bert_config,num_labels=6)
# net.load_state_dict(torch.load("../input/bert-model5/bert_pytorch_v5.pt"))
# net.cuda()

# ## inference
# net.eval()
# result_3 = list()
# with torch.no_grad():
#   for (x_batch,) in test_loader:
#     y_pred = net(x_batch)
#     y_pred = torch.sigmoid(y_pred.cpu()).numpy()[:,0]
#     result_3.extend(y_pred)
# result_3 = np.array(result_3)

net = BertForSequenceClassification(bert_config,num_labels=6)
net.load_state_dict(torch.load("../input/bert-model6/bert_pytorch_v6.pt"))
net.cuda()

## inference
net.eval()
result_4 = list()
with torch.no_grad():
  for (x_batch,) in test_loader:
    y_pred = net(x_batch)
    y_pred = torch.sigmoid(y_pred.cpu()).numpy()[:,0]
    result_4.extend(y_pred)
result_4 = np.array(result_4)


net = BertForSequenceClassification(bert_config,num_labels=6)
net.load_state_dict(torch.load("../input/bert-model7/bert_pytorch.pt"))
net.cuda()

## inference
net.eval()
result_5 = list()
with torch.no_grad():
  for (x_batch,) in test_loader:
    y_pred = net(x_batch)
    y_pred = torch.sigmoid(y_pred.cpu()).numpy()[:,0]
    result_5.extend(y_pred)
result_5 = np.array(result_5)

net = BertForSequenceClassification(bert_config,num_labels=6)
net.load_state_dict(torch.load("../input/bert-model8/bert_pytorch.pt"))
net.cuda()

## inference
net.eval()
result_6 = list()
with torch.no_grad():
  for (x_batch,) in test_loader:
    y_pred = net(x_batch)
    y_pred = torch.sigmoid(y_pred.cpu()).numpy()[:,0]
    result_6.extend(y_pred)
result_6 = np.array(result_6)


## LSTM part

## read data
test_df = pd.read_csv("../input/jigsaw-unintended-bias-in-toxicity-classification/test.csv")
test_df['comment_text'] = test_df['comment_text'].fillna('_##_').values

## preprocessing
x_test = test_df["comment_text"].apply(lambda x: content_preprocessing(x))


## Tokenize and padding
tokenizer = text.Tokenizer(filters='', lower=False)
tokenizer.fit_on_texts(list(x_test))

x_test = tokenizer.texts_to_sequences(x_test)
x_test = sequence.pad_sequences(x_test, maxlen=MAX_LEN,padding='post')

#### build DataLoader
x_test_cuda = torch.tensor(x_test, dtype=torch.long).cuda()
test_data = torch.utils.data.TensorDataset(x_test_cuda)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

## build word embedding
crawl_matrix ,_ = build_matrix(tokenizer.word_index, CRAWL_EMBEDDING_PATH)


## load pretrained model
net = NeuralNet(crawl_matrix, 128)
model_dict = net.state_dict()
pretrained_dict = torch.load("../input/lstm-model2/rnn_pytorch.pt")
del pretrained_dict['embedding.weight']
model_dict.update(pretrained_dict) 
net.load_state_dict(model_dict)
net.cuda()


## inference
net.eval()
result_rnn_1 = list()
with torch.no_grad():
	for (x_batch,) in test_loader:
		y_pred, _ = net(x_batch)
		y_pred = y_pred.cpu().numpy()[:,0]
		result_rnn_1.extend(y_pred)

result_rnn_1 = np.array(result_rnn_1)

## load pretrained model
net = NeuralNet(crawl_matrix, 128)
model_dict = net.state_dict()
pretrained_dict = torch.load("../input/lstm-model3/rnn_pytorch.pt")
del pretrained_dict['embedding.weight']
model_dict.update(pretrained_dict) 
net.load_state_dict(model_dict)
net.cuda()


## inference
net.eval()
result_rnn_2 = list()
with torch.no_grad():
	for (x_batch,) in test_loader:
		y_pred, _ = net(x_batch)
		y_pred = y_pred.cpu().numpy()[:,0]
		result_rnn_2.extend(y_pred)

result_rnn_2 = np.array(result_rnn_2)

## load pretrained model
net = NeuralNet(crawl_matrix, 128)
model_dict = net.state_dict()
pretrained_dict = torch.load("../input/lstm-model6/rnn_pytorch.pt")
del pretrained_dict['embedding.weight']
model_dict.update(pretrained_dict) 
net.load_state_dict(model_dict)
net.cuda()


## inference
net.eval()
result_rnn_3 = list()
with torch.no_grad():
	for (x_batch,) in test_loader:
		y_pred, _ = net(x_batch)
		y_pred = y_pred.cpu().numpy()[:,0]
		result_rnn_3.extend(y_pred)

result_rnn_3 = np.array(result_rnn_3)

## submission
submission = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/test.csv', index_col='id')
submission = submission.drop(columns=['comment_text'])
submission['prediction'] = (result_rnn_1*0.33+result_rnn_2*0.33+result_rnn_3*0.34)*0.3+(result_1*0.2+result_2*0.2+result_4*0.2+result_5*0.2+result_6*0.2)*0.7
submission.reset_index(drop=False, inplace=True)
submission.to_csv('submission.csv', index=False)
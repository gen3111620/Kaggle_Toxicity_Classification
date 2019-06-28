# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

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
from tqdm import tqdm
tqdm.pandas()
from gensim.models import KeyedVectors
from flashtext import KeywordProcessor

CRAWL_EMBEDDING_PATH = '../input/fasttext-crawl-300d-2m/crawl-300d-2M.vec'
# CRAWL_EMBEDDING_PATH = '../input/paragram-dandrocec/paragram_300_sl999.txt'
# GLOVE_EMBEDDING_PATH = '../input/glove840b300dtxt/glove.840B.300d.txt'

BATCH_SIZE = 1024
EPOCHS = 5
MAX_LEN = 220
NUM_MODEL = 1
seed = 9635

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

def custom_loss(data, targets):

    ''' Define custom loss function for weighted BCE on 'target' column '''
    bce_loss = nn.BCELoss(weight=targets[:,1])(data[:,0],targets[:,0])
    return bce_loss
    

# SWEAR_WORDS_PATH = '../input/entxt/en.txt'
# SWEAR_WORDS_PATH = '../input/badword/en.txt'

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
    

def statistics_upper_words(text):
    upper_count = 0
    for token in text.split():
        if re.search(r'[A-Z]', token):
            upper_count += 1
    return upper_count

def statistics_unique_words(text):
    words_set = set()

    for token in text.split():
        words_set.add(token)

    return len(words_set)

def statistics_characters_nums(text):

    chars_set = set()

    for char in text:
        chars_set.add(char)
    
    return len(chars_set)

def statistics_swear_words(text):
    swear_count = 0
    for swear_word in swear_words:
        if swear_word in text:
            swear_count += 1
    return swear_count

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
    
    # upper_count = statistics_upper_words(text)
    # characters_num = statistics_characters_nums(text)
    # unique_words_num = statistics_unique_words(text)
    # swear_words_num = statistics_swear_words(text)

            
    # return text, swear_words_num, len(text.split()), emoji_num, upper_count, unique_words_num, characters_num
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
		self.embedding_dropout = SpatialDropout(0.3)
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
		
		
train_df = pd.read_csv("../input/jigsaw-unintended-bias-in-toxicity-classification/train.csv")
test_df = pd.read_csv("../input/jigsaw-unintended-bias-in-toxicity-classification/test.csv")
train_df['comment_text'] = train_df['comment_text'].fillna('_##_').values
test_df['comment_text'] = test_df['comment_text'].fillna('_##_').values

# shuffling the data
train_df = train_df.sample(frac=1).reset_index(drop=True)

print("Train shape : ", train_df.shape)
print("Test shape : ", test_df.shape)


### preprocessing
# x_train, x_train_swear_words, x_train_token_num, x_train_emoji_num, x_train_upper_count, x_train_unique_words_num, x_train_characters_num = zip(*train_df["comment_text"].apply(lambda x: content_preprocessing(x)))
x_train = train_df["comment_text"].apply(lambda x: content_preprocessing(x))

# x_train_swear_words = np.array(x_train_swear_words, dtype=np.long).reshape(-1, 1)
# x_train_token_num = np.array(x_train_token_num, dtype=np.long).reshape(-1, 1)
# x_train_emoji_num = np.array(x_train_emoji_num, dtype=np.long).reshape(-1, 1)
# x_train_upper_count = np.array(x_train_upper_count, dtype=np.long).reshape(-1, 1)
# x_train_unique_words_num = np.array(x_train_unique_words_num, dtype=np.long).reshape(-1, 1)
# x_train_characters_num = np.array(x_train_characters_num, dtype=np.long).reshape(-1, 1)

y_aux_train = np.array(train_df[['severe_toxicity', 'obscene', 'identity_attack', 'insult', 'threat']])

# x_test, x_test_swear_words, x_test_token_num, x_test_emoji_num, x_test_upper_count, x_test_unique_words_num, x_test_characters_num  = zip(*test_df["comment_text"].apply(lambda x: content_preprocessing(x)))
x_test = test_df["comment_text"].apply(lambda x: content_preprocessing(x))

# x_test_swear_words = np.array(x_test_swear_words).reshape(-1, 1)
# x_test_token_num = np.array(x_test_token_num, dtype=np.long).reshape(-1, 1)
# x_test_emoji_num = np.array(x_test_emoji_num, dtype=np.long).reshape(-1, 1)
# x_test_upper_count = np.array(x_test_upper_count, dtype=np.long).reshape(-1, 1)
# x_test_unique_words_num = np.array(x_test_unique_words_num, dtype=np.long).reshape(-1, 1)
# x_test_characters_num = np.array(x_test_characters_num, dtype=np.long).reshape(-1, 1)

# del x_train_swear_words, x_train_token_num,x_train_emoji_num, x_train_upper_count, x_train_unique_words_num, x_train_characters_num
# del x_test_swear_words, x_test_token_num,x_test_emoji_num, x_test_upper_count, x_test_unique_words_num, x_test_characters_num
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
# max_features = 400000
tokenizer = text.Tokenizer(filters='', lower=False)


tokenizer.fit_on_texts(list(x_train)+list(x_test))

x_train = tokenizer.texts_to_sequences(x_train)
x_test = tokenizer.texts_to_sequences(x_test)

x_train = sequence.pad_sequences(x_train, maxlen=MAX_LEN,padding='post')
x_test = sequence.pad_sequences(x_test, maxlen=MAX_LEN,padding='post')


# shuffling the data
# np.random.seed()
train_idx = np.random.permutation(len(x_train))
x_train = x_train[train_idx]
y_train = y_train[train_idx]

#### Set model parameters
x_test_cuda = torch.tensor(x_test, dtype=torch.long).cuda()
test_data = torch.utils.data.TensorDataset(x_test_cuda)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

del x_test_cuda


crawl_matrix ,_ = build_matrix(tokenizer.word_index, CRAWL_EMBEDDING_PATH)
# glove_matrix ,_ = build_matrix(tokenizer.word_index, GLOVE_EMBEDDING_PATH)
# embedding_matrix = (crawl_matrix+glove_matrix)/2
# embedding_matrix = np.concatenate((crawl_matrix, glove_matrix), axis=-1)

# del crawl_matrix, glove_matrix
final_test = list()

for index in range(NUM_MODEL):
    
    
	seed_everything(seed+index)
	x_train_fold, x_val, y_train_fold, y_val = train_test_split(x_train, y_train, test_size=0.1)

	x_train_fold = torch.tensor(x_train, dtype=torch.long).cuda()
	x_val = torch.tensor(x_val, dtype=torch.long).cuda()
	y_train_fold = torch.tensor(y_train, dtype=torch.float).cuda()
	y_val = torch.tensor(y_val, dtype=torch.float).cuda()

	train_data = torch.utils.data.TensorDataset(x_train_fold, y_train_fold)
	val_data = torch.utils.data.TensorDataset(x_val, y_val)

	train_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
	val_loader = torch.utils.data.DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=True)

	print("model: {}".format(index))
	net = NeuralNet(crawl_matrix, 128)
	net.cuda()
	loss_fn = torch.nn.BCELoss(reduction='mean')
	optimizer = torch.optim.Adam(net.parameters(), lr=0.002)

	test_checkpoint = list()
	loss_checkpoint = list()

	for epoch in range(EPOCHS):  # loop over the dataset multiple times

		start_time = time.time()

		avg_loss = 0.0

		net.train()
		for i, data in enumerate(train_loader):

			# get the inputs
			inputs, labels = data

			label1 = labels[:,:2]
			label2 = labels[:,2:]

			## forward + backward + optimize
			pred1, pred2 = net(inputs)

			loss1 =	custom_loss(pred1, label1)
			loss2 = loss_fn(pred2,label2)
			loss = loss1*loss_weight+loss2

			# zero the parameter gradients
			optimizer.zero_grad()

			loss.backward()
			optimizer.step()

			avg_loss += loss.item()

		avg_val_loss = 0.0
		net.eval()
		for data in val_loader:

			# get the inputs
			inputs, labels = data

			val_label1 = labels[:,:2]
			val_label2 = labels[:,2:]

			## forward + backward + optimize
			pred1, pred2 = net(inputs)

			loss1_val = custom_loss(pred1, val_label1)
			# loss2_val = loss2 = loss_fn(pred2,val_label2)
			# loss_val = loss1_val*loss_weight+loss2_val

			avg_val_loss += loss1_val.item()

		elapsed_time = time.time() - start_time 

		print('Epoch {}/{} \t loss={:.4f}\t val_loss={:.4f} \t time={:.2f}s'.format(
						epoch+1, EPOCHS, avg_loss/len(train_loader),avg_val_loss/len(val_loader), elapsed_time))
		
		# print('Epoch {}/{} \t loss={:.4f} \t time={:.2f}s'.format(
		# 				epoch+1, EPOCHS, avg_loss/len(train_loader), elapsed_time))

		## inference
		net.eval()
		result = list()
		with torch.no_grad():
			for (x_batch,) in test_loader:
				y_pred, _ = net(x_batch)
				y_pred = y_pred.cpu().numpy()[:,0]
				result.extend(y_pred)

		test_checkpoint.append(result)
		# loss_checkpoint.append(avg_val_loss)

	final_test.append(test_checkpoint[-1])
	torch.save(net.state_dict(), "rnn_pytorch.pt")
    



## submission
submission = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/sample_submission.csv', index_col='id')
submission['prediction'] = np.mean(final_test, axis=0)
submission.reset_index(drop=False, inplace=True)
submission.to_csv('submission.csv', index=False)
#!/usr/bin/env python
# coding=utf-8
import os
import logging
import pickle
import random
import spacy
import gensim
from gensim.utils import simple_preprocess
from nltk.stem import WordNetLemmatizer, SnowballStemmer
stemmer = SnowballStemmer("english")
import numpy as np
import re
from bs4 import BeautifulSoup
from flashtext import KeywordProcessor

# import nltk
# nltk.download('wordnet')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def seed_everything(seed=6089):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)


seed_everything()


class text_preprocessing():
    def __init__(self, corpus):
        '''Implement several text preprocessing functions.
        :param corpus: raw data of a piece of corpus
        :type corpus: str
        '''
        self.corpus = str(corpus)
        self.punctuations = {'》', '〞', '¢', '‹', '╦', '║', '♪', 'Ø', '╩', '\\', '★', '＋', 'ï', '<', '?', '％', '+', '„', 'α', '*', '〰', '｟', '¹', '●', '〗', ']', '▾', '■', '〙', '↓', '´', '【', 'ᴵ',
                             '"', '）', '｀', '│', '¤', '²', '‡', '¿', '–', '」', '╔', '〾', '%', '¾', '←', '〔', '＿', '’', '-', ':', '‧', '｛', 'β', '（', '─', 'à', 'â', '､', '•', '；', '☆', '／', 'π', 'é', '╗', '＾', '▪',
                             ',', '►', '/', '〚', '¶', '♦', '™', '}', '″', '＂', '『', '▬', '±', '«', '“', '÷', '×', '^', '!', '╣', '▲', '・', '░', '′', '〝', '‛', '√', ';', '】', '▼', '.', '~', '`', '。', 'ə', '］', '，',
                             '{', '～', '！', '†', '‘', '﹏', '═', '｣', '〕', '〜', '＼', '▒', '＄', '♥', '〛', '≤', '∞', '_', '[', '＆', '→', '»', '－', '＝', '§', '⋅', '▓', '&', 'Â', '＞', '〃', '|', '¦', '—', '╚', '〖',
                             '―', '¸', '³', '®', '｠', '¨', '‟', '＊', '£', '#', 'Ã', "'", '▀', '·', '？', '、', '█', '”', '＃', '⊕', '=', '〟', '½', '』', '［', '$', ')', 'θ', '@', '›', '＠', '｝', '¬', '…', '¼', '：',
                             '¥', '❤', '€', '−', '＜', '(', '〘', '▄', '＇', '>', '₤', '₹', '∅', 'è', '〿', '「', '©', '｢', '∙', '°', '｜', '¡', '↑', 'º', '¯', '♫'}

        self.mispelled_dict = {'colour': 'color', 'centre': 'center', 'favourite': 'favorite', 'travelling': 'traveling', 'counselling': 'counseling', 'theatre': 'theater', 'cancelled': 'canceled', 'labour': 'labor', 'organisation': 'organization', 'wwii': 'world war 2', 'citicise': 'criticize', 'youtu ': 'youtube ', 'qoura': 'quora', 'sallary': 'salary', 'Whta': 'What', 'narcisist': 'narcissist', 'howdo': 'how do', 'whatare': 'what are', 'howcan': 'how can', 'howmuch': 'how much', 'howmany': 'how many', 'em': 'them', 'whydo': 'why do', 'doI': 'do I', 'theBest': 'the best', 'howdoes': 'how does', 'mastrubation': 'masturbation', 'mastrubate': 'masturbate', 'mastrubating': 'masturbating', 'pennis': 'penis', 'etherium': 'ethereum', 'narcissit': 'narcissist', 'bigdata': 'big data', '2k17': '2017', '2k18': '2018', '2k19': '2019', 'qouta': 'quota', 'exboyfriend': 'ex boyfriend', 'airhostess': 'air hostess', 'whst': 'what', 'watsapp': 'whatsapp', 'demonitisation': 'demonetization', 'demonitization': 'demonetization', 'demonetisation': 'demonetization', 'pokémon': 'pokemon', 'n*gga': 'nigga', 'p*': 'pussy', 'b***h': 'bitch', 'a***h****': 'asshole', 'a****le-ish': 'asshole', 'b*ll-s***': 'bullshit', 'd*g': 'dog', 'st*up*id': 'stupid', 'd***': 'dick', 'di**': 'dick', "ain't": 'is not', "aren't": 'are not', "can't": 'cannot', "'cause": 'because', "could've": 'could have', "couldn't": 'could not', "didn't": 'did not', "doesn't": 'does not', "don't": 'do not', "hadn't": 'had not', "hasn't": 'has not', "haven't": 'have not', "he'd": 'he would', "he'll": 'he will', "he's": 'he is', "how'd": 'how did', "how'd'y": 'how do you', "how'll": 'how will', "how's": 'how is', "I'd": 'I would', "I'd've": 'I would have', "I'll": 'I will', "I'll've": 'I will have', "I'm": 'I am', "I've": 'I have', "i'd": 'i would', "i'd've": 'i would have', "i'll": 'i will', "i'll've": 'i will have', "i'm": 'i am', "i've": 'i have', "isn't": 'is not', "it'd": 'it would', "it'd've": 'it would have', "it'll": 'it will', "it'll've": 'it will have', "it's": 'it is', "let's": 'let us', "ma'am": 'madam', "mayn't": 'may not', "might've": 'might have', "mightn't": 'might not', "mightn't've": 'might not have', "must've": 'must have', "mustn't": 'must not', "mustn't've": 'must not have', "needn't": 'need not', "needn't've": 'need not have', "o'clock": 'of the clock', "oughtn't": 'ought not', "oughtn't've": 'ought not have', "shan't": 'shall not', "sha'n't": 'shall not', "shan't've": 'shall not have', "she'd": 'she would', "she'd've": 'she would have', "she'll": 'she will', "she'll've": 'she will have', "she's": 'she is', "should've": 'should have', "shouldn't": 'should not', "shouldn't've": 'should not have', "so've": 'so have', "so's": 'so as', "this's": 'this is', "that'd": 'that would', "that'd've": 'that would have', "that's": 'that is', "there'd": 'there would', "there'd've": 'there would have', "there's": 'there is', "here's": 'here is', "they'd": 'they would', "they'd've": 'they would have', "they'll": 'they will', "they'll've": 'they will have', "they're": 'they are', "they've": 'they have', "to've": 'to have', "wasn't": 'was not', "we'd": 'we would', "we'd've": 'we would have', "we'll": 'we will', "we'll've": 'we will have', "we're": 'we are', "we've": 'we have', "weren't": 'were not', "what'll": 'what will', "what'll've": 'what will have', "what're": 'what are', "what's": 'what is', "what've": 'what have', "when's": 'when is', "when've": 'when have', "where'd": 'where did', "where's": 'where is', "where've": 'where have', "who'll": 'who will', "who'll've": 'who will have', "who's": 'who is', "who've": 'who have', "why's": 'why is', "why've": 'why have', "will've": 'will have', "won't": 'will not', "won't've": 'will not have', "would've": 'would have', "wouldn't": 'would not', "wouldn't've": 'would not have', "y'all": 'you all', "y'all'd": 'you all would', "y'all'd've": 'you all would have', "y'all're": 'you all are', "y'all've": 'you all have', "you'd": 'you would', "you'd've": 'you would have', "you'll": 'you will', "you'll've": 'you will have', "you're": 'you are', "you've": 'you have', 'Qoura': 'Quora', 'Etherium': 'Ethereum', '\u200b': ' ', '\ufeff': '', 'करना': '', 'है': '', 'sh*tty': 'shitty', 's**t': 'shit', 'nigg*r': 'nigger', 'bulls**t': 'bullshit', 'n*****': 'nigger', 'p*ssy': 'pussy', 'p***y': 'pussy', 'f***': 'fuck', 'f*^k': 'fuck', 'f*cked': 'fucked', 'f*ck': 'fuck', 'f***ing': 'fucking', 'F*CKING': 'fucking', 'sh*t': 'shit', 'su*k': 'suck', 'a**holes': 'assholes', 'a**hole': 'asshole', 'di*k': 'dick', 'd*ck': 'dick', 'd**k': 'dick', 'bull**it': 'bullshit', 'c**t': 'cunt', 'cu*t': 'cunt', 'c*nt': 'cunt', 'troʊl': 'trool', 'trumpian': 'bombast', 'realdonaldtrump': 'trump', 'drumpf': 'trump', 'trumpist': 'trump', "i'ma": 'i am', "is'nt": 'is not', '‘I': 'I', 'ᴀɴᴅ': 'and', 'ᴛʜᴇ': 'the', 'ʜᴏᴍᴇ': 'home', 'ᴜᴘ': 'up', 'ʙʏ': 'by', 'ᴀᴛ': 'at', '…and': 'and', 'civilbeat': 'civil beat', 'TrumpCare': 'Trump care', 'Trumpcare': 'Trump care', 'OBAMAcare': 'Obama care', 'ᴄʜᴇᴄᴋ': 'check', 'ғᴏʀ': 'for', 'ᴛʜɪs': 'this', 'ᴄᴏᴍᴘᴜᴛᴇʀ': 'computer', 'ᴍᴏɴᴛʜ': 'month', 'ᴡᴏʀᴋɪɴɢ': 'working', 'ᴊᴏʙ': 'job', 'ғʀᴏᴍ': 'from', 'Sᴛᴀʀᴛ': 'start', 'gubmit': 'submit', 'CO₂': 'carbon dioxide', 'ғɪʀsᴛ': 'first', 'ᴇɴᴅ': 'end', 'ᴄᴀɴ': 'can', 'ʜᴀᴠᴇ': 'have', 'ᴛᴏ': 'to', 'ʟɪɴᴋ': 'link', 'ᴏғ': 'of', 'ʜᴏᴜʀʟʏ': 'hourly', 'ᴡᴇᴇᴋ': 'week', 'ᴇxᴛʀᴀ': 'extra', 'Gʀᴇᴀᴛ': 'great', 'sᴛᴜᴅᴇɴᴛs': 'student', 'sᴛᴀʏ': 'stay', 'ᴍᴏᴍs': 'mother', 'ᴏʀ': 'or', 'ᴀɴʏᴏɴᴇ': 'anyone', 'ɴᴇᴇᴅɪɴɢ': 'needing', 'ᴀɴ': 'an', 'ɪɴᴄᴏᴍᴇ': 'income', 'ʀᴇʟɪᴀʙʟᴇ': 'reliable', 'ʏᴏᴜʀ': 'your', 'sɪɢɴɪɴɢ': 'signing', 'ʙᴏᴛᴛᴏᴍ': 'bottom', 'ғᴏʟʟᴏᴡɪɴɢ': 'following', 'Mᴀᴋᴇ': 'make', 'ᴄᴏɴɴᴇᴄᴛɪᴏɴ': 'connection', 'ɪɴᴛᴇʀɴᴇᴛ': 'internet', 'financialpost': 'financial post', 'ʜaᴠᴇ': ' have ', 'ᴄaɴ': ' can ', 'Maᴋᴇ': ' make ', 'ʀᴇʟɪaʙʟᴇ': ' reliable ', 'ɴᴇᴇᴅ': ' need ', 'ᴏɴʟʏ': ' only ', 'ᴇxᴛʀa': ' extra ', 'aɴ': ' an ', 'aɴʏᴏɴᴇ': ' anyone ', 'sᴛaʏ': ' stay ', 'Sᴛaʀᴛ': ' start', 'SHOPO': 'shop', 'ᴀ': 'A', 'theguardian': 'the guardian', 'deplorables': 'deplorable', 'theglobeandmail': 'the globe and mail', 'justiciaries': 'justiciary', 'creditdation': 'Accreditation', 'doctrne': 'doctrine', 'fentayal': 'fentanyl', 'designation-': 'designation', 'CONartist': 'con-artist', 'Mutilitated': 'Mutilated', 'Obumblers': 'bumblers', 'negotiatiations': 'negotiations', 'dood-': 'dood', 'irakis': 'iraki', 'cooerate': 'cooperate', 'COx': 'cox', 'racistcomments': 'racist comments', 'envirnmetalists': 'environmentalists', 'SB91': 'senate bill', 'tRump': 'trump', 'utmterm': 'utm term', 'FakeNews': 'fake news', 'Gʀᴇat': 'great', 'ʙᴏᴛtoᴍ': 'bottom', 'washingtontimes': 'washington times', 'garycrum': 'gary crum', 'htmlutmterm': 'html utm term', 'RangerMC': 'car', 'TFWs': 'tuition fee waiver', 'SJWs': 'social justice warrior', 'Koncerned': 'concerned', 'Vinis': 'vinys', 'Yᴏᴜ': 'you', 'trumpists': 'trump', 'trumpkins': 'trump', 'trumpism': 'trump', 'trumpsters': 'trump', 'thedonald': 'trump', 'trumpty': 'trump', 'trumpettes': 'trump', 'trumpland': 'trump', 'trumpies': 'trump', 'trumpo': 'trump', 'drump': 'trump', 'dtrumpview': 'trump', 'drumph': 'trump', 'trumpanzee': 'trump', 'trumpite': 'trump', 'chumpsters': 'trump', 'trumptanic': 'trump', 'itʻs': 'it is', 'donʻt': 'do not', 'pussyhats': 'pussy hats', 'trumpdon': 'trump', 'trumpisms': 'trump', 'trumperatti': 'trump', 'legalizefreedom': 'legalize freedom', 'trumpish': 'trump', 'ur': 'you are', 'twitler': 'twitter', 'trumplethinskin': 'trump', 'trumpnuts': 'trump', 'trumpanzees': 'trump', 'justmaybe': 'just maybe', 'trumpie': 'trump', 'trumpistan': 'trump', 'trumphobic': 'trump', 'piano2': 'piano', 'trumplandia': 'trump', 'globalresearch': 'global research', 'trumptydumpty': 'trump', 'frank1': 'frank', 'trumpski': 'trump', 'trumptards': 'trump', 'alwaysthere': 'always there', 'clickbait': 'click bait', 'antifas': 'antifa', 'dtrump': 'trump', 'trumpflakes': 'trump flakes', 'trumputin': 'trump putin', 'fakesarge': 'fake sarge', 'civilbot': 'civil bot', 'tumpkin': 'trump', 'trumpians': 'trump', 'drumpfs': 'trump', 'dtrumpo': 'trump', 'trumpistas': 'trump', 'trumpity': 'trump', 'trump nut': 'trump', 'russiagate': 'russia gate', 'trumpsucker': 'trump sucker', 'trumpbart': 'trump bart', 'trumplicrat': 'trump', 'dtrump0': 'trump', 'tfixstupid': 'stupid', 'brexit': 'British exit', 'Brexit': 'British exit', 'trumpelthinskin': 'trump', 'americanophobia': 'Anti-Americanism', 'magaphants': 'anti-trump', 'MAGAphants': 'anti-trump', 'klastri': '<b>', 'cheetolini': 'trump', 'daesh': 'ISIS'}

    @staticmethod
    def stem(text):
        return stemmer.stem(text)

    @staticmethod
    def lemmatize_word(word, pov):
        return WordNetLemmatizer().lemmatize(word, pos=pov)

    @staticmethod
    def lemmatize(text):
        # Initialize spacy 'en' model, keeping only tagger component needed for lemmatization
        nlp = spacy.load('en', disable=['parser', 'ner'])
        # Parse the sentence using the loaded 'en' model object `nlp`
        doc = nlp(text)
        # Extract the lemma for each token and join
        return " ".join([token.lemma_ for token in doc])

    @staticmethod
    def strip_tags_uris(text):
        uri_re = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        #uri_re = r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))'
        # BeautifulSoup on content
        soup = BeautifulSoup(text, "html.parser")
        # Stripping all <code> tags with their content if any (note: other tags such as <script> could be removed using this function)
        if soup.code:
            soup.code.decompose()
        # Get all the text out of the html
        text = soup.get_text()
        # Returning text stripping out all uris
        return re.sub(uri_re, "url", text)

    @staticmethod
    def strip_emoji(text):
        emoji_re = re.compile(u'['
                              u'\U00010000-\U0010ffff'
                              u'\U0001F600-\U0001F64F'
                              u'\U0001F300-\U0001F5FF'
                              u'\U0001F30D-\U0001F567'
                              u'\U0001F680-\U0001F6FF'
                              u'\u2122-\u2B55]', re.UNICODE)
        return emoji_re.sub('', text)

    def strip_punctuation(self, text):
        for punct in self.punctuations:
            text = text.replace(punct, f' {punct} ')
        return text

    def fix_mispelled_dict(self, text):
        kp = KeywordProcessor(case_sensitive=True)
        mix_mispelled_dict = {}
        for k, v in self.mispelled_dict.items():
            mix_mispelled_dict[k] = v
            mix_mispelled_dict[k.lower()] = v.lower()
            mix_mispelled_dict[k.upper()] = v.upper()
            mix_mispelled_dict[k.capitalize()] = v.capitalize()

        for k, v in mix_mispelled_dict.items():
            kp.add_keyword(k, v)
        return kp.replace_keywords(text)

    @staticmethod
    def lowercase(text):
        return text.lower()

    @staticmethod
    def strip_stopwords(text):
        spacy_nlp = spacy.load('en_core_web_sm')
        spacy_stopwords = spacy.lang.en.stop_words.STOP_WORDS
        return ' '.join([t for t in text.split() if t not in spacy_stopwords])

    @staticmethod
    def strip_digits(text):
        return re.sub(r'\d+', '', text)

    @staticmethod
    def strip_control_char(text):
        text = re.sub(r'\n\r\t', ' ', text)
        text = re.sub(r'\s{2,}', ' ', text)
        return text

    def simple_preprocess(self, text):
        result = []
        text = self.stem(self.lemmatize(text))
        for token in gensim.utils.simple_preprocess(text):
            if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
                result.append(self.stem(self.lemmatize(token)))
        return result

    def custom_preprocess(self):

        processed_text = self.lowercase(self.corpus)
        processed_text = self.strip_emoji(processed_text)
        # processed_text = self.strip_tags_uris(processed_text)
        processed_text = self.fix_mispelled_dict(processed_text)
        processed_text = self.strip_punctuation(processed_text)
        processed_text = self.stem(processed_text)
        processed_text = self.lemmatize(processed_text)
        processed_text = self.strip_stopwords(processed_text)
        processed_text = self.strip_digits(processed_text)
        processed_text = self.strip_control_char(processed_text)

        return processed_text


if __name__ == '__main__':
    s = '''Target on russian dbo

https://virustotal.com/ru/file/6cdff1af ... /analysis/
https://virustotal.com/ru/file/8cd25454 ... 465199488/
User avatar
Username
Ludvig
Posts
7
Joined
Fri Jan 29, 2016 9:30 am
Re: Bolik /bankbot,infector 32 and 64/
 #28626  by EP_X0FF
 Mon Jun 06, 2016 12:21 pm
VT links posting are great, now attach the actual files or your post makes absolutely no sense.

Ring0 - the source of inspiration'''
    test = text_preprocessing(s)
    print(test.custom_preprocess())

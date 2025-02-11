# pip install nlp-id
# pip install sastrawi
# pip install pyngrok
# pip install flask-cors
# pip install numpy==1.23.5 
# pip scikit-learn==1.2.2 imbalanced-learn==0.10.1

import pandas as pd
import numpy as np
import seaborn as sns
import nltk
import string
import re
import json
import pickle
import joblib
import re
import requests
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import joblib
import pickle
import joblib
import re
import requests
from dateutil import parser
from datetime import datetime
from collections import Counter
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from nlp_id.lemmatizer import Lemmatizer
from nlp_id.tokenizer import Tokenizer
from nlp_id.tokenizer import PhraseTokenizer
from nlp_id.stopword import StopWord
from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn.pipeline import Pipeline
from sklearn.utils import _param_validation
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from flask import Flask, request, jsonify
from flask_cors import CORS
from pyngrok import ngrok
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from flask import Flask, request, jsonify
from flask_cors import CORS
from pyngrok import ngrok
nltk.download('punkt')

factory = StopWordRemoverFactory()
stopword = factory.create_stop_word_remover()
print(joblib.__version__)

def casefolding(text):
  return text.lower()
def filtering(text):
  # Remove link web
  text = re.sub(r'http\S+', '', text)
  # Remove @username
  text = re.sub('@[^\s]+', '', text)
  # Remove #tagger
  text = re.sub(r'#([^\s]+)', '', text)
  # Remove angka termasuk angka yang berada dalam string
  # Remove non ASCII chars
  text = re.sub(r'[^\x00-\x7f]', r'', text)
  text = re.sub(r'(\\u[0-9A-Fa-f]+)', r'', text)
  text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
  text = re.sub(r'\\u\w\w\w\w', '', text)
  # Remove simbol, angka dan karakter aneh
  text = re.sub(r"[.,:;+!\-_<^/=?\"'\(\)\d\*]", " ", text)
  text = re.sub(r"\bADVERTISEMENT\b", "", text, flags=re.IGNORECASE)
  text = re.sub(r"\bSCROLL TO CONTINUE WITH CONTENT\b", "", text, flags=re.IGNORECASE)
  text = re.sub(r"\bBACA\b", "", text, flags=re.IGNORECASE)
  text = re.sub(r"\bBACA SELENGKAPNYA\b", "", text, flags=re.IGNORECASE)
  text = re.sub(r"\bHALAMAN\b", "", text, flags=re.IGNORECASE)
  text = re.sub(r"\bHALAMAN SELANJUTNYA\b", "", text, flags=re.IGNORECASE)
  text = re.sub(r"\bHALAMAN BERIKUTNYA\b", "", text, flags=re.IGNORECASE)
  text = re.sub(r"\bCOM\b", "", text, flags=re.IGNORECASE)
  text = re.sub(r"\bHTTP\b", "", text, flags=re.IGNORECASE)
  text = re.sub(r"\bCO\b", "", text, flags=re.IGNORECASE)
  return text
def replaceThreeOrMore(text):
  # Pattern to look for three or more repetitions of any character, including newlines (contoh goool -> gool).
  pattern = re.compile(r"(.)\1{1,}", re.DOTALL)
  return pattern.sub(r"\1\1", text)
def removeDoubleSpaces(text):
  while '  ' in text:
    text = text.replace('  ', ' ')
  return text


import requests

# Unduh daftar kata slang
url = 'https://raw.githubusercontent.com/louisowen6/NLP_bahasa_resources/master/combined_slang_words.txt'
kamus_slangword = requests.get(url).text
dict_slang = eval(kamus_slangword)

# Hapus entri "pp" dari kamus slang
if 'pp' in dict_slang:
    del dict_slang['pp']
elif 'kk' in dict_slang:
    del dict_slang['kk']

def convertToSlangword(text):
    text = text.split()
    content = []
    for kata in text:
        if kata in dict_slang:
            kata = dict_slang[kata]
        content.append(kata)
    return ' '.join(content)

def getConversions(text):
    text = text.split()
    conversions = []
    for kata in text:
        if kata in dict_slang:
            conversions.append((kata, dict_slang[kata]))
    return conversions


def preprocess_text(text):
    text = casefolding(text)
    text = filtering(text)
    text = replaceThreeOrMore(text)
    text = removeDoubleSpaces(text)
    text = convertToSlangword(text)
    return text


factory = StemmerFactory()
stemmer = factory.create_stemmer()
stop_factory = StopWordRemoverFactory()

# Fungsi untuk tokenisasi
def tokenizeText(text):
    token = nltk.word_tokenize(text)
    return token

def removeStopWords(tokens):
    stopwords = set(stop_factory.get_stop_words())
    filtered_tokens = [word for word in tokens if word.lower() not in stopwords]
    return filtered_tokens

# Fungsi untuk stemming
def stemText(tokens):
    # Mengembalikan list token yang telah distem
    stemmed_tokens = [stemmer.stem(word) for word in tokens]
    return stemmed_tokens

app = Flask(__name__)
CORS(app)

def casefolding(text):
    return text.lower()

def filtering(text):
    # (Fungsi filtering seperti yang Anda berikan)
    return text

def replaceThreeOrMore(text):
    pattern = re.compile(r"(.)\1{1,}", re.DOTALL)
    return pattern.sub(r"\1\1", text)

def removeDoubleSpaces(text):
    while '  ' in text:
        text = text.replace('  ', ' ')
    return text

# Unduh daftar kata slang
url = 'https://raw.githubusercontent.com/louisowen6/NLP_bahasa_resources/master/combined_slang_words.txt'
kamus_slangword = requests.get(url).text
dict_slang = eval(kamus_slangword)
if 'pp' in dict_slang:
    del dict_slang['pp']
elif 'kk' in dict_slang:
    del dict_slang['kk']

def convertToSlangword(text):
    text = text.split()
    content = []
    for kata in text:
        if kata in dict_slang:
            kata = dict_slang[kata]
        content.append(kata)
    return ' '.join(content)

def preprocess_text(text):
    text = casefolding(text)
    text = filtering(text)
    text = replaceThreeOrMore(text)
    text = removeDoubleSpaces(text)
    text = convertToSlangword(text)
    return text

def text_process2(text):
    clean_text = preprocess_text(text)
    tokens = tokenizeText(clean_text)
    stopwords = removeStopWords(tokens)
    stemmed_tokens = stemText(stopwords)
    return ' '.join(stemmed_tokens)

# Load model
svm_model_path = './model/svm_svmsmote_randomize_model_hoax_detection.pkl'
model_random_search_svm_smote = joblib.load(svm_model_path)

# Definisikan route untuk Flask
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = data.get('text', '')
    processed_text = text_process2(text)
    prediction_svm = model_random_search_svm_smote.predict([processed_text])
    probability_svm = model_random_search_svm_smote.predict_proba([processed_text])
    return jsonify({
        "original_text": text,
        "preprocessed_text": processed_text,
        "prediction": "HOAX" if prediction_svm[0] == 1 else "VALID",
        "hoax_probability": round(probability_svm[0][1] * 100, 2),
        "valid_probability": round(probability_svm[0][0] * 100, 2),
    })

ngrok.set_auth_token('2kN8waXfVvvfRZETIlkSw6r1rH7_56b3WhYToR4YQEKMKjTRX')   

# Setup ngrok and run Flask
public_url = ngrok.connect(5000)
print('Akses aplikasi di URL:', public_url)

app.run(port=5000)


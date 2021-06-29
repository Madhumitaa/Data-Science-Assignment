import spacy
import en_core_web_lg
import json
import nltk
import re
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pandas as pd
import pickle


nlp = en_core_web_lg.load()


def read_json():
    Article_list=[]
    with open('Article_stack.json', 'r', encoding='utf8') as json_file:
        data = json.load(json_file)
  
    # for reading nested data [0] represents
    # the index value of the list
    for k,v in data.items():
           Article_list.append(str(v))
           text=""
           text = text.join(Article_list)
    extract_noun_chunks(text)
          

def extract_noun_chunks(text1):
    doc = nlp(text1)
    sentence = list(doc.sents)
    ents = list(doc.ents)
    nouns = []
    for np in doc.noun_chunks:
        nouns.append(np.text)
    noun_chunck_model(nouns)

def noun_chunck_model(n):
    vectorizer = TfidfVectorizer(ngram_range = (2,3))
    X2 = vectorizer.fit_transform(n)
    scores = (X2.toarray())
    features = (vectorizer.get_feature_names())
    sums = X2.sum(axis = 0)
    data1 = []
    for col, term in enumerate(features):
        data1.append( (term, sums[0,col] ))
    ranking = pd.DataFrame(data1, columns = ['term','rank'])
    words = (ranking.sort_values('rank', ascending = False))
    a_list = words['term'].tolist()
    result = []
    for i in range(10):
        result.append(a_list[i])
    print(result)

read_json()

vectorizer = TfidfVectorizer(ngram_range = (2,3))
pickle.dump(vectorizer, open('top_nc_model.pkl','wb'))

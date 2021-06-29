import nltk
import re
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pandas as pd
import json

Article_list=[]

  
# Opening JSON file
with open('Article_stack.json', 'r', encoding='utf8') as json_file:
    data = json.load(json_file)
  
    # for reading nested data [0] represents
    # the index value of the list
    for k,v in data.items():
        print(str(v))
        Article_list.append(str(v))

def preprocess(sent):
    sent = nltk.word_tokenize(sent)
    sent = nltk.pos_tag(sent)
    return sent
np=[]

for i in range(len(Article_list)):
  sent = preprocess(Article_list[i])
  pattern = 'NP: {<DT>?<JJ>*<NN>}'
  cp = nltk.RegexpParser(pattern)
  cs = cp.parse(sent)
  for subtree in cs.subtrees(filter=lambda t: t.label() == 'NP'):
    n=[x[0] for x in subtree.leaves()]
    s=" "
    x=s.join(n)
    np.append(x)
#removing duplicates    
ns=set(np)

noun_chunks = list(ns)

#using tf idf vectorizer

vectorizer = TfidfVectorizer(ngram_range = (2,3))
X2 = vectorizer.fit_transform(noun_chunks)
scores = (X2.toarray())
print("\n\nScores : \n", scores)

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









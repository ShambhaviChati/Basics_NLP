# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 14:10:32 2020

@author: Shambhavi Chati
"""

import pandas as pd
import numpy as np
import re
import sklearn
from sklearn import model_selection
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import f1_score
from sklearn import preprocessing, decomposition, model_selection, metrics, pipeline
from sklearn.model_selection import GridSearchCV,StratifiedKFold,RandomizedSearchCV
import nltk

from collections import Counter
from itertools import chain
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.util import ngrams
from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english")
from bs4 import BeautifulSoup
lem = WordNetLemmatizer()
from nltk.corpus import stopwords
stop = stopwords.words('english')
import string



data_train = pd.read_csv("train.csv")
data_test = pd.read_csv("test.csv")
#Removing location variable
data_train.drop('location', axis= 1, inplace = True )
def remove_URL(text):
    url = re.compile(r'https?://\S+|www\.\S+')
    return url.sub(r'',text)
data_train['text']=data_train['text'].apply(lambda x : remove_URL(x))
data_test['text']=data_test['text'].apply(lambda x : remove_URL(x))
#data cleaning steps
def remove_punct(df):
    df["text"] = df["text"].str.replace(r'https?://[^\s<>"]+|www\.[^\s<>"]+', "")
    df["text"] = df['text'].str.replace('[^\w\s]','')
    df["text"] = df['text'].str.lower()
    
    return df
data_train = remove_punct(data_train)
data_test = remove_punct(data_test)

def nltk_processing(df):
    df['text_without_stopwords'] = df['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
    mask = [isinstance(item, (str, bytes)) for item in df['text_without_stopwords']]
    df = df.loc[mask]
    df['tokenised'] = df['text_without_stopwords'].apply(word_tokenize)
    df=  df.drop('text_without_stopwords', axis=1)
    return df

data_train = nltk_processing(data_train)
data_test = nltk_processing(data_test)

def bigrams(df):
    df['bigrams'] = df['tokenised'].apply(lambda row: list(nltk.ngrams(row, 2)))
    df['trigrams'] = df['tokenised'].apply(lambda row: list(nltk.ngrams(row, 3)))
    trigrams = df['trigrams'].tolist()
    bigrams = df['bigrams'].tolist()
    #counting bigrams frequency
    bigrams = list(chain(*bigrams))
    bigrams = [(x.lower(), y.lower()) for x,y in bigrams]
    bigram_counts = Counter(bigrams)
    frequent_bigrams = bigram_counts.most_common(100) 
    print(frequent_bigrams)
    return(df)
   
data_train = bigrams(data_train)
data_test = bigrams(data_test)

#Transfroming words to vectors
count_vectorizer = CountVectorizer()
train_vectors = count_vectorizer.fit_transform(data_train['text'])
test_vectors = count_vectorizer.transform(data_test["text"])


#TF-IDF vectorisation to remove dominance of words with higher frequency
tfidf = TfidfVectorizer(min_df=2, max_df=0.5, ngram_range=(1, 2))
train_tfidf = tfidf.fit_transform(data_train['text'])
test_tfidf = tfidf.transform(data_test["text"])   
 
# Developin a Logistic Regression Model

#Sigmoid function

# Fitting a simple Logistic Regression on Counts
log = LogisticRegression(C=1.0)
scores = model_selection.cross_val_score(log, train_vectors, data_train["target"], cv=5, scoring="f1")
print(scores)
log.fit(train_vectors, data_train['target'])

# Fitting a simple Logistic Regression on TFIDF
clf_tfidf = LogisticRegression(C=1.0)
scores = model_selection.cross_val_score(clf_tfidf, train_tfidf, data_train["target"], cv=5, scoring="f1")
print(scores)

#Naive Bayes
clf_NB = MultinomialNB()
scores = model_selection.cross_val_score(clf_NB, train_tfidf, data_train["target"], cv=5, scoring="f1")
clf_NB.fit(train_tfidf, data_train["target"])
print(scores)
 

def submission(submission_file_path,model,test_vectors):
    sample_submission = pd.read_csv(submission_file_path)
    sample_submission["target"] = model.predict(test_vectors)
    sample_submission.to_csv("submission.csv", index=False)   
    
submission_file_path = "D:/Shambhavi Chati/Projects/Kaggle/nlp-getting-started/sample_submission.csv"
test_vectors=test_tfidf
submission(submission_file_path,clf_NB,test_vectors)
    
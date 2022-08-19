import sys
# adding to the path variables the one folder higher (locally, not changing system variables)
sys.path.append("..")

# importing all needed libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import nltk
from nltk.corpus import stopwords
# from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
import multiprocessing
from sklearn import utils

import time
from tqdm import tqdm

# ignore the warnings
import warnings
warnings.filterwarnings("ignore")

# set Randomseed
RSEED = 42

# import needed functions
from modeling.processing import *

# load the first 100k lines of the review file into a dataframe
dfr = pd.read_csv('../data/yelp_dataset/review_1819.csv')
print('Data set imported.')

# filter for only english reviews
dfr = language_processing(dfr, verbose=True)
# initialize the stopword list:
stopwords = nltk.corpus.stopwords.words('english')
# remove punctuation from the text in the initial df
dfr['text'] = dfr['text'].apply(remove_punctuation)
print('Dataset cleaned.')

# split data into train and test set
train_set, test_set = train_test_split(pd.concat([dfr['text'], dfr['stars']], axis=1), random_state=RSEED) # concat…

# building mostly on (and partly copied from) https://towardsdatascience.com/multi-class-text-classification-with-doc2vec-logistic-regression-9da9947b43f4
# documentation: https://radimrehurek.com/gensim/models/doc2vec.html
def tokenize(text):
    tokens = []
    for sent in nltk.sent_tokenize(text):
        for token in nltk.word_tokenize(sent):
            tokens.append(token)
    return tokens

def vec_for_learning(model, tagged_docs):
    sents = tagged_docs.values
    # targets, regressors = zip(*[(doc.tags[0], model.infer_vector(doc.words, steps=20)) for doc in sents]) # TODO check importance of "steps" argument
    targets, regressors = zip(*[(doc.tags[0], model.infer_vector(doc.words)) for doc in sents])
    return targets, regressors

train_tagged = train_set.apply(lambda r: TaggedDocument(words=tokenize(r.text), tags=[r.stars]), axis=1)
print('Training dataset tokenized.')

# d2v = Doc2Vec(dm=0, vector_size=300, negative=5, hs=0, min_count=2, sample=0, workers=multiprocessing.cpu_count())
d2v = Doc2Vec(dm=0, vector_size=100, negative=5, hs=0, min_count=2, sample=0, workers=multiprocessing.cpu_count())
d2v.build_vocab([x for x in tqdm(train_tagged)])
print('Vocab built.')

print('Start training:')
# for epoch in range(30):
for epoch in range(10):
    d2v.train(utils.shuffle([x for x in tqdm(train_tagged)]), total_examples=len(train_tagged), epochs=1)
    d2v.alpha -= 0.002
    d2v.min_alpha = d2v.alpha
print('Training done.')

model_fn = '../models/doc2vec_100.model'
d2v.save(model_fn)
print('Model saved to "{model_fn}".')
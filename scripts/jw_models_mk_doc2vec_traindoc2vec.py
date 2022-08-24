# importing all needed libraries
import multiprocessing
import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn import utils
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from tqdm import tqdm
import string
from datetime import datetime

import sys
# adding to the path variables the one folder higher (locally, not changing system variables)
sys.path.append("..")

# import needed functions
from processing import *

# ignore the warnings
import warnings
warnings.filterwarnings("ignore")

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

def print_w_timestamp(text):
    print(f"{datetime.now().time().strftime('%H:%M:%S.%f')}\t{text}")


# set Randomseed
RSEED = 42
DATA_FN_IN = '../data/yelp_dataset/review_1819.csv'
# MODEL_FN = '../models/doc2vec_100.model' # vect. size 100, min_count 100, 10 epochs, alpha 0.002
# MODEL_FN = '../models/doc2vec_300_small.model' # vect. size 300, min_count 2, 10 epochs, alpha 0.002
# MODEL_FN = '../models/doc2vec_300_small_30epochs.model' # vect. size 300, min_count 2, 30 epochs, alpha 0.002
# MODEL_FN = '../models/doc2vec_300_small_5epochs.model' # vect. size 300, min_count 2, 5 epochs, alpha 0.002
MODEL_FN = '../models/doc2vec_500_small_5epochs.model' # vect. size 300, min_count 2, 5 epochs, alpha 0.002


# load the review file into a dataframe
print_w_timestamp('Start importing data set…')
dfr = pd.read_csv(DATA_FN_IN).iloc[:200000]
print_w_timestamp('Data set imported.')

# filter for only english reviews
print_w_timestamp('Start restricting data set to English reviews…')
dfr = language_processing(dfr)
print_w_timestamp('Data set restricted to English reviews.')

# initialize the stopword list:
stopwords = nltk.corpus.stopwords.words('english')
# remove punctuation from the text in the initial df
print_w_timestamp('Start cleaning data set…')
# dfr['text'] = dfr['text'].apply(remove_punctuation)
dfr['text'] = dfr['text'].apply(lambda s: s.translate(str.maketrans('', '', string.punctuation)))
print_w_timestamp('Dataset cleaned.')

# split data into train and test set
train_set, test_set = train_test_split(pd.concat([dfr['text'], dfr['stars']], axis=1), random_state=RSEED) # TODO concat…
print_w_timestamp('Dataset split in train and test set.')

print_w_timestamp('Start tokenizing training dataset…')
train_tagged = train_set.apply(lambda r: TaggedDocument(words=tokenize(r.text), tags=[r.stars]), axis=1)
print_w_timestamp('Training dataset tokenized.')

print_w_timestamp('Start building vocabulary…')
# d2v = Doc2Vec(dm=0, vector_size=300, negative=5, hs=0, min_count=2, sample=0, workers=multiprocessing.cpu_count())
# d2v = Doc2Vec(dm=0, vector_size=100, negative=5, hs=0, min_count=2, sample=0, workers=multiprocessing.cpu_count())
# d2v = Doc2Vec(dm=0, vector_size=100, negative=5, hs=0, min_count=100, sample=0, workers=multiprocessing.cpu_count())
d2v = Doc2Vec(dm=0, vector_size=500, negative=5, hs=0, min_count=2, sample=0, workers=multiprocessing.cpu_count())
d2v.build_vocab(train_tagged)
print_w_timestamp('Vocabulary built.')

print_w_timestamp('Start training…')
for epoch in range(5):
# for epoch in range(30):
# for epoch in range(10):
    print_w_timestamp(f"Start epoch {epoch+1}…")
    d2v.train(utils.shuffle(train_tagged), total_examples=len(train_tagged), epochs=1)
    d2v.alpha -= 0.002
    d2v.min_alpha = d2v.alpha
    print_w_timestamp(f"Epoch {epoch+1} done.")
print_w_timestamp('Training done.')

d2v.save(MODEL_FN)
print_w_timestamp(f"Model saved to \"{MODEL_FN}\".")
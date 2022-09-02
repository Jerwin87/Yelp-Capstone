import pandas as pd
# from collections import defaultdict
from gensim import corpora, models
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
# import string

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

import sys
# adding to the path variables the one folder higher (locally, not changing system variables)
sys.path.append("..")
from scripts.processing import *
import datetime

# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import confusion_matrix, classification_report


def print_w_time(text):
    print(f"{datetime.datetime.now().strftime('%H:%M:%S.%f')}\t{text}")

def tok(document_text, stoplist):
    # return [[ token for token in word_tokenize(document_text.translate(str.maketrans('', '', string.punctuation)).lower()) ] if token not in stoplist]
    return [ token for token in word_tokenize(document_text.lower()) if token not in stoplist]

def train_LDA_model(df, begin=None, end=None, num_topics=40, passes=1, iterations=50):
    print_w_time("Start preprocessing…")
    # TODO check further possible preprocessing steps
    df = language_processing(df, verbose=False)
    print_w_time("Language processing done.")

    # https://radimrehurek.com/gensim/auto_examples/core/run_topics_and_transformations.html#sphx-glr-auto-examples-core-run-topics-and-transformations-py
    # https://towardsdatascience.com/end-to-end-topic-modeling-in-python-latent-dirichlet-allocation-lda-35ce4ed6b3e0
    documents = []
    for document in df.iloc[begin:end].text:
        documents.append(document)

    # TODO use different "tokenizer"?
    stoplist = stopwords.words('english')
    texts = [tok(document, stoplist) for document in documents]
    print_w_time("Tokenization done.")

    # https://radimrehurek.com/gensim/auto_examples/tutorials/run_lda.html
    # Remove numbers, but not words that contain numbers.
    texts = [[token for token in doc if not token.isnumeric()] for doc in texts]
    # Remove words that are only one character.
    texts = [[token for token in doc if len(token) > 1] for doc in texts]
    print_w_time("Numbers and words with one character removed.")

    lemmatizer = WordNetLemmatizer()
    texts = [ [lemmatizer.lemmatize(token) for token in doc] for doc in texts ]
    print_w_time("Lemmatization done.")

    # (slightly adapted) from https://radimrehurek.com/gensim/auto_examples/tutorials/run_lda.html
    # Add bigrams and trigrams to docs (only ones that appear 20[now 10] times or more).
    # TODO check if something can be improved (not really improving anything like this…)
    bigram = models.Phrases(texts, min_count=10)
    for idx in range(len(texts)):
        for token in bigram[texts[idx]]:
            if '_' in token:
                # Token is a bigram, add to document.
                texts[idx].append(token)
    print_w_time("n-grams done.")

    # freq = defaultdict(int)
    # for text in texts:
    #     for token in text:
    #         freq[token] += 1

    # texts = [[token for token in text if freq[token] > 1] for text in texts]
    # print_w_time("Infrequent words removed.")

    dictionary = corpora.Dictionary(texts)

    # https://radimrehurek.com/gensim/auto_examples/tutorials/run_lda.html
    # Filter out words that occur less than 20 documents, or more than 50% of the documents.
    dictionary.filter_extremes(no_below=20, no_above=0.5)

    corpus = [dictionary.doc2bow(text) for text in texts]
    # tfidf = models.TfidfModel(corpus)
    # corpus_tfidf = tfidf[corpus]
    print_w_time("Texts vectorized.")
    # TODO check other vectorization methods

    print_w_time("Start training of LDA model…")
    lda_model = models.LdaMulticore(corpus, id2word=dictionary, num_topics=num_topics,
                                    passes=passes, iterations=iterations, eval_every=1)
    # TODO consider more hyperparameters!
    print_w_time("LDA model trained.")
    return lda_model, dictionary

def add_topics(df, lda_model, dictionary, begin=None, end=None, num_topics=40):
    print_w_time("Start adding topics…")
    stoplist = stopwords.words('english')
    lemmatizer = WordNetLemmatizer()
    rows_topics = []
    i = 0
    total = len(df.iloc[begin:end])
    for _, row in df.iloc[begin:end].iterrows():
        row_text_tok_lem = [lemmatizer.lemmatize(token) for token in tok(row.text, stoplist)]
        row_doc = dictionary.doc2bow(row_text_tok_lem)
        dict_topics = {} 
        for topic in lda_model[row_doc]:
            dict_topics[f"topic_{topic[0]}"] = 1 if topic[1] > 0 else 0 # ????
        dict_topics['review_id'] = row.review_id
        rows_topics.append(dict_topics)
        i += 1
        if i % 5000 == 0:
            print_w_time(f"{i/total*100: 3.1f}%")
    print_w_time("Topics added.")
    return pd.DataFrame(rows_topics).fillna(0)


if __name__ == '__main__':
    # RSEED = 42
    # PASSES = 10
    # ITERATIONS = 200
    # df = pd.read_csv('../data/yelp_dataset/review_1819.csv')
    # print_w_time("DataFrame read.")
    # for num_topics in range(10, 101, 10):
    #     print(f"Start training with num_topics={num_topics}…")
    #     lda_model, dictionary = train_LDA_model(df, end=100000, num_topics=num_topics,
    #                                             passes=PASSES, iterations=ITERATIONS)
    #     X = add_topics(df, lda_model, dictionary, begin=200000, end=300000, num_topics=num_topics)
    #     y = df.iloc[200000:300000]['useful'].apply(lambda x: 1 if x > 0 else 0)
    #     
    #     # split data into train and test set
    #     X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=RSEED, stratify=y)

    #     # initialize the Classifier
    #     logreg = LogisticRegression(max_iter=10000)

    #     # fit the model
    #     logreg.fit(X_train, y_train)    
    #     
    #     # make predictions
    #     y_pred = logreg.predict(X_test)
    #     
    #     # test the model
    #     print(f"\n\nRESULTS for training with num_topics={num_topics}:")
    #     print(classification_report(y_test, y_pred))
    #     print(confusion_matrix(y_test, y_pred))
    #     print("\n\n")

    RSEED = 42
    NUM_TOPICS = 40
    df = pd.read_csv('../data/yelp_dataset/review_1819.csv')
    # lda_model, dictionary = train_LDA_model(df, end=100000, num_topics=NUM_TOPICS)
    lda_model, dictionary = train_LDA_model(df, end=500000, num_topics=NUM_TOPICS)
    # lda_model, dictionary = train_LDA_model(df, num_topics=NUM_TOPICS)
    # df_topics = add_topics(df, lda_model, dictionary, begin=200000, end=300000, num_topics=NUM_TOPICS)
    df_topics = add_topics(df, lda_model, dictionary, num_topics=NUM_TOPICS)
    df_merged = df.merge(df_topics, on='review_id', how='left').fillna(0)
    df_merged.to_csv('review_1819_topics_train500000_annotfull.csv')

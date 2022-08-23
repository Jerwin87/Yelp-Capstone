import sys
sys.path.append('..')

# doc: https://radimrehurek.com/gensim/models/word2vec.html
from gensim.models import Word2Vec

from modeling.processing import get_df


DATA_FN_IN = '../data/yelp_dataset/yelp_academic_dataset_business.json'
MODEL_FN = "../models/cat_word2vec.model"


# import dataframe
df = get_df(DATA_FN_IN, limit=None)

# get and split categories
cats = df.categories.dropna().values # TODO What form do the NAs have? Check!
cats_separated = [ [token.strip() for token in item.lower().split(',')] for item in cats] # remove commas in a different way?

# train and save word2vec model
model = Word2Vec(sentences=cats_separated, vector_size=100, window=3, min_count=1, workers=4)
model.save(MODEL_FN)

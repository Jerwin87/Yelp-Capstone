import sys
sys.path.append('..')

import numpy as np

# doc: https://radimrehurek.com/gensim/models/word2vec.html
from gensim.models import Word2Vec

from modeling.processing import get_df


DATA_FN_IN = '../data/yelp_dataset/yelp_academic_dataset_business.json'
DATA_FN_OUT = '../data/yelp_dataset/business_maincatadded.csv'
MODEL_FN = "../models/cat_word2vec.model"


# define main categories
# TODO add more categories (cf. notebooks/mk_categories.ipynb) and perhaps also
# a second level of categories
# TODO find most abstract/(?) distinct categories
# TODO possibly also: combine later to more abstract categories
main_cats = ['restaurants', 'shopping', 'religious organizations', 'home services',
 'health & medical', 'local services', 'hotels & travel', 'arts & entertainment',
 'fitness & instruction', 'bakeries', 'grocery', 'beauty & spas', 'automotive']

# load word2vec model
model = Word2Vec.load(MODEL_FN)

# import business dataset
df = get_df(DATA_FN_IN, limit=None)

# assign main cat. to each row
# TODO search for items that are very distant from the next main category and
# find better main categories for these cases.
cats_business_maincatadded = []
for cats_business in df.categories.values:
    # ignore "None"
    # TODO check rows with None value
    if cats_business == None:
        cats_business_maincatadded.append('')
        continue
    # get word vector for each category in category list
    cats_business_vectors = []
    for cat in [token.strip() for token in cats_business.lower().split(',')]:
        cats_business_vectors.append(model.wv[cat])
    # calculate mean of word vectors
    cats_business_vectors_mean = np.mean(cats_business_vectors, axis=0)
    # find closest main cat. to mean of word vectors
    most_similar_main_cat = main_cats[
        np.argmin(model.wv.distances(cats_business_vectors_mean, main_cats))]
    # append main cat. assigned to row
    cats_business_maincatadded.append(most_similar_main_cat)

# add main cat. for rows as new column to dataframe
df['maincat'] = cats_business_maincatadded

# save business dataset with main cat. added
df.to_csv(DATA_FN_OUT)

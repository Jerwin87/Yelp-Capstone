import pandas as pd

# doc: https://radimrehurek.com/gensim/models/word2vec.html
from gensim.models import Word2Vec

import sys
sys.path.append('..')
from modeling.processing import get_df

def train():
    df = get_df('/Users/mkinzler/neuefische/Yelp-Capstone/data/yelp_academic_dataset_business.json', limit=None)
    cats = df.categories.dropna().values # What form do the NAs have?
    
    cats_separated = [ [token.strip() for token in item.lower().split(',')] for item in cats] # remove commas in a different way?
    # print(cats_separated[0:5])
    
    model = Word2Vec(sentences=cats_separated, vector_size=100, window=3, min_count=1, workers=4)
    model.save("cat_word2vec.model")

# train()

model = Word2Vec.load("cat_word2vec.model")
# print(model.wv.key_to_index)

### # adapted from https://towardsdatascience.com/google-news-and-leo-tolstoy-visualizing-word2vec-word-embeddings-with-t-sne-11558d8bd4d
### keys = ['restaurants', 'breweries', 'shopping', 'doctors']
### embedding_clusters = []
### word_clusters = []
### 
### for word in keys:
###     embeddings = []
###     similiar_words = []
###     for sim_word, _ in model.wv.most_similar(word, topn=20):
###         similiar_words.append(sim_word)
###         embeddings.append(model.wv[sim_word])
###     word_clusters.append(similiar_words)
###     embedding_clusters.append(embeddings)
###         
### # print(similiar_words)
### from sklearn.manifold import TSNE
### import numpy as np
### 
### embedding_clusters = np.array(embedding_clusters)
### n, m, k = embedding_clusters.shape
### print(n, m, k)
### tsne_model_2d = TSNE(perplexity=15, n_components=2, init='pca', n_iter=3500, random_state=42)
### embeddings_2d = np.array(tsne_model_2d.fit_transform(embedding_clusters.reshape(n*m, k))).reshape(n, m, 2)
### 
### import matplotlib.pyplot as plt
### import matplotlib.cm as cm
### from mpl_toolkits.mplot3d import Axes3D
### 
### 
### a = 0.7
### plt.figure(figsize=(16, 9))
### colors = cm.rainbow(np.linspace(0, 1, len(keys)))
### for label, embeddings, words, color in zip(keys, embeddings_2d, word_clusters, colors):
###     x = embeddings[:, 0]
###     y = embeddings[:, 1]
###     plt.scatter(x, y, c=color, alpha=a, label=label)
###     for i, word in enumerate(words):
###         plt.annotate(word, alpha=0.5, xy=(x[i], y[i]), xytext=(5, 2),
###                      textcoords='offset points', ha='right', va='bottom', size=8)
### plt.savefig('cat_word2vec.png', format='png', dpi=150, bbox_inches='tight')
### plt.show()
### 
### # tsne_model_3d = TSNE(perplexity=30, n_components=3, init='pca', n_iter=3500, random_state=42)
### # # embeddings_3d = tsne_model_3d.fit_transform(embedding_clusters)
### # embeddings_3d = tsne_model_3d.fit_transform(embedding_clusters.reshape(n*m, k)).reshape(n, m, 3)
### # 
### # a = 0.5
### # fig = plt.figure()
### # ax = Axes3D(fig)
### # colors = cm.rainbow(np.linspace(0, 1, 1))
### # 
### # plt.scatter(embeddings_3d[:, 0], embeddings_3d[:, 1], embeddings_3d[:, 2], c=colors, alpha=a)
### # for i, word in enumerate(words):
### #     ax.text(x[i], y[i], z[i], word, size=8, zorder=1, color=color)
### # # for label, embeddings, words, color in zip(keys, embeddings_3d, word_clusters, colors):
### # #     x = embeddings[:, 0]
### # #     y = embeddings[:, 1]
### # #     z = embeddings[:, 2]
### # #     ax.scatter(x, y, z, c=color, alpha=a, label=label)
### # #     # for i, word in enumerate(words):
### # #     #     ax.text(x[i], y[i], z[i], word, size=8, zorder=1, color=color)
### # 
### # plt.show()

# clustering K-means
# inspirated by https://dylancastillo.co/nlp-snippets-cluster-documents-using-word2vec/
import numpy as np

df = get_df('/Users/mkinzler/neuefische/Yelp-Capstone/data/yelp_academic_dataset_business.json', limit=None)
cats = df.categories.dropna().values # What form do the NAs have?
    
cats_separated = [ [token.strip() for token in item.lower().split(',')] for item in cats] # remove commas in a different way?

doc_vectors_mean = []

for doc in cats_separated[0:100]:
    doc_vectors = []
    for token in doc:
        doc_vectors.append(model.wv[token])
    doc_vectors_mean.append(np.mean(doc_vectors, axis=0)) # we also need some labels…

from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_samples, silhouette_score

K = 5
km = MiniBatchKMeans(n_clusters=K, batch_size=50).fit(doc_vectors_mean)
silhouette_samples_values = silhouette_samples(doc_vectors_mean, km.labels_)
for i in range(K):
    cluster_sv = silhouette_samples_values[km.labels_ == i]
    print(i, cluster_sv.shape[0], cluster_sv.mean(), cluster_sv.min(), cluster_sv.max())



# import matplotlib.pyplot as plt
# import matplotlib.cm as cm
# from mpl_toolkits.mplot3d import Axes3D
# from sklearn.decomposition import PCA
# 
# # https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html#sklearn.decomposition.PCA
# pca = PCA(n_components=2, random_state=42)
# embeddings_2d_centers = pca.fit_transform(km.cluster_centers_)
# pca = PCA(n_components=2, random_state=42)
# embeddings_2d_all = pca.fit_transform(doc_vectors_mean)
# print(embeddings_2d_all)
# 
# a = 0.7
# plt.figure(figsize=(16, 9))
# colors = cm.rainbow(np.linspace(0, 1, 3))
# # for embeddings, color in zip(embeddings_2d_centers, colors):
# #      x = embeddings[0]
# #      y = embeddings[1]
# #      plt.scatter(x, y, c=color, alpha=a)
# import seaborn as sns
# sns.scatterplot(data=embeddings_2d_all)
# ### plt.savefig('cat_word2vec.png', format='png', dpi=150, bbox_inches='tight')
# plt.show()

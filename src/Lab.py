"""
Gensim Library - one of many core NLP libraries
Used to:
  1. Retrieval
  2. Topic modelling
  3. Representation Learning (word2vec and doc2vec)

  """

# Import libraries

import gensim
import numpy as np
import pandas as pd

import gensim.downloader as api
from gensim import utils
from gensim.models import KeyedVectors
from gensim.test.utils import datapath
from gensim.models import Word2Vec

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import LabelEncoder

from scipy.stats import pearsonr, spearmanr

import nltk
from nltk.corpus import stopwords

import torch
import torch.nn as nn

import plotly.express as px

"""
By calling an API we tell the gensim which embeddings we want

output:

The object we get is of type KeyedVectors

"""

word_emb = api.load('word2vec-google-news-300')


"""
Let's show how those embeddings look like in 2 ways:

1. Assess the embeddings with word-lookup
2. Assess the embeddings with index lookup

"""

# First way

print(word_emb["apple"])

# Second way

print(word_emb[10])

"""
Let's check the vocabulary.

Two important attributes:

1. key_to_index: maps a word to its vocabulary index
2. index_to_key: maps a vocabulary index to corresponding word

"""

# From the vocabulary index we want to get the vocabulary word 

print(f"Vocabulary length {len(word_emb.key_to_index)}")
print(f"Index of cat {word_emb.key_to_index['cat']}")
print(f"Word at position 654 {word_emb.index_to_key[654]}")

"""
Compute similarity and distance

Given a list of pairs of words find the cosine similarity between them, 
obtaining fixed the first word

"""

pairs = [
('car','minivan'),
('car','bicycle'),
('car','airplane'),
('car','cereal'),
('car','communism')
]

print("w1   w2   cos_sim   cos_dist")
for w1, w2 in pairs:
    print(f"{w1}  {w2}  {word_emb.similarity(w1, w2):.3f}  {word_emb.distance(w1, w2):.3f}")

"""
Nearest Neighbour Retrieval

"""    

def retrieve_most_similar(query_words, all_word_emb, restrict_vocab = 10000):

    #Step 1: Get full or restricted vocabulary embeddings

    vocab_emb = all_word_emb.vectors[:restrict_vocab+1,:] if restrict_vocab is not None else all_word_emb.vectors

    #Step 2: get the word embeddings for the query words

    query_emb = all_word_emb[query_words]

    #Step 3:get cosine similarity between queries and embeddings

    cos_sim = cosine_similarity(query_emb, vocab_emb)

    #Step 4: Sort similarities in desceding orders and get indices of nearest neighbours

    nn = np.argsort(-cos_sim)

    #Step 5: delete self similarity, i.e. cos_sim(w,w) = 1.0

    nn_filtered = nn[:, 1:]
    
    #Step 6: use the indisces to get the words

    nn_words = np.array(word_emb.index_to_key)[nn_filtered]

    return nn_words
   

# test the function

queries = ["king","queen","italy","Italy","nurse"]
res = retrieve_most_similar(queries, word_emb, restrict_vocab=10000)
top_k = 10
res_k = res[:, :top_k]
del res
print(res_k)

# Dimensionality reduction and plotting

all_res_words = res_k.flatten()
res_word_emb = word_emb[all_res_words]
print("(|Q| x k) x word_emb_size")
print(res_word_emb.shape)

# Perform 3D-PCA

pca = PCA(n_components = 3)
word_emb_pca = pca.fit_transform(res_word_emb)

pca_df = pd.DataFrame(word_emb_pca, columns=["pca_x","pca_y","pca_z"])

pca_df["word"] = res_k.flatten()

labels = np.array([queries]).repeat(top_k)
pca_df["query"] = labels

print(pca_df.head())

px.scatter_3d (pca_df, x='pca_x', y='pca_y', z='pca_z', color="query", text="word", opacity=0.7, title="3d-PCA representation of word embeddings")

"""
Word embedding evaluation:

1. intrinsic evaluation: evaluate embedding without a downstream taks

   a. word similarity benchmarks
   b. word analogy benchmarks

2. extrinsic evaluation: evaluate word embeddings on a downstream tast

"""

"""
Word similarity benchmarks, such as WS353, contain word pairs and human- given similarity score

"""

ws353_df = pd.read_csv(datapath('wordsim353.tsv'), sep="\t", skiprows=1).rename(columns={"# Word 1": "Word 1"})
ws353_df.sample(5)

"""
Three steps to evaluate word embeddings:

1. For every pair in our dataset we get the embeddings
2. For ach pair we compute cosine similarity between its word embeddings,
   we call the similarity function
3. Then, we simply compute the correlation score, even Pearson's r or Spearman's p
   between the human given score h and the cosine similarity s

!! Gensim provides us with a function: evaluate_word_pairs

"""

word_emb.evaluate_word_pairs(datapath('wordsim353.tsv'), case_insensitive = False)

"""
Word analogy benchmarks

man : king = woman : x

word2vec paper shows that word2vec embeddings can solve (some) of these equations by algebric operations:

Get  ex=eking−eman+ewoman 
Check if  NNV(ex)=queen


Gensim provides us with a most_similar function
It has several arguments, the most important are:
positive : list of words that should be summed together
negative : list of words that should be subtracted


"""

print(word_emb.most_similar(positive=["king", "woman"], negative=["man"], restrict_vocab=100000))
print(word_emb.most_similar(positive=["iPod", "Sony"], negative=["Apple"], restrict_vocab=100000))

f = open(datapath('questions-words.txt'))
print("".join(f.readlines()[:15]))
f.close()

accuracy, results = word_emb.evaluate_word_analogies(datapath('questions-words.txt'))
print(f"Accuracy {accuracy}")
print(results[0].keys())
print(f"Correct {results[0]['correct'][:5]}")
print(f"Inorrect {results[0]['incorrect'][:5]}")

"""
Implement intrisic evaluation uwingwordsim353 benchmark

"""

# reload the data

ws353_df = pd.read_csv(datapath('wordsim353.tsv'), sep = "\t", skiprows=1).rename(columns ={"# Word 1": "Word 1"})

# Get embeddings

w1 = word_emb[ws353_df["Word 1"]]
w2 = word_emb[ws353_df["Word 2"]]

# compute cosine similarities

cos_mat = cosine_similarity(w1, w2)
cos_pairs = np.diag(cos_mat)

# compute correlations

print(pearsonr(cos_pairs, ws353_df["Human (mean)"]))
print(spearmanr(cos_pairs, ws353_df["Human (mean)"]))

"""
Pretraining your own embeddings

"""
# get a dataset

corpus = open(datapath('lee_background.cor'))
sample = corpus.readline()
print(sample, utils.simple_preprocess(sample))


class MyCorpus:
    """ An iterator that yields sentences """

    def __iter__(self):
        corpus_path = datapath('lee_background.cor')
        for line in open(corpus_path):
            # we assume there's one document per line, tokens separated by whitespace
            yield utils.simple_preprocess(line)



# pretrain our own embeddings
# we will use the Word2Vec class from gensim.models
# Vocabulary building + training 

model = Word2Vec(sentences = MyCorpus(),
                 min_count = 3, # ignore all words with freq < min_count
                 vector_size = 200, # dimensionality of the vectors
                 sg = 1, # set to 1 for skip-gram
                 epochs = 10,
                 alpha = 0.025, # initial learning rate
                 batch_words = 10000, # batch size
                 window = 5, # window size for context words
                 negative = 10, # number of negatives for negative sampling
                 ns_exponent = 0.75 # exponent of the sampling distribution
                )

print(model)

word_emb_lee = model.wv # wv attribute contains word embeddings

"""
Saving and loading embeddings

Saving and loading the full model (embeddings plus hyperparameters)
allows us to resume training

If you save only word embeddings, this does not allow to resume training
"""

save_path = "word2vec_lee.model"
model.save(save_path)
model_reloaded = Word2Vec.load(save_path)

save_path = "word2vee_lee.emb"
model.wv.save(save_path)
emb_reloaded = KeyedVectors.load(save_path)

"""
Extrinsic evaluation of word embeddings
In this example, we will use them to solve a spam classification task

"""

spam_df = pd.read_csv("data/SMSSpamCollection.tsv", sep="\t", header=None, names=["label", "text"])
spam_df
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 23:17:42 2021

@author: ASUS
"""
import json
with open("sikm_groupsio-l.json", 'rb') as f:
    bstream = f.read()
txt = bstream.decode("utf-8")

import re
txt_proper2 = re.split('\r\n',txt)


min_length_content = 10;

# fix json non standard format https://www.json.org/json-en.html
posts_list = []
for i, ln in enumerate(txt_proper2):
    if len(ln) > 0:
        to_append = json.loads(ln)
        if len(to_append["content"]) > min_length_content:
            # forget about comment at this moment
            to_append_content = to_append['content']
            posts_list.append(to_append_content)

                   
                   
import nltk
from nltk.corpus import stopwords




def is_valid_posts(p):
    if len(nltk.tokenize(p)) < min_length_content :
        return False
    
    return True

stopwords_list =  stopwords.words("english")

posts_list_tokenized = [nltk.tokenize.word_tokenize(p) for p in posts_list]

posts_list_tokenized_stop_removed = []

#remove stopword and punctuation


   

#%%    
# simplex case, monogram

post_corpus=set()
from collections import Counter
post_corpus_counter = Counter()
ndoc_counter = Counter()
def filter_word(list_token):
    return [word.lower() for word in list_token if word.isalpha() and  word not in stopwords_list]
    
doc_req_counter = Counter()
gram = 1
if gram == 1:
    for p in posts_list_tokenized:
        p2 = filter_word(p)
        if len(p2) >= min_length_content:
            posts_list_tokenized_stop_removed.append(p2 )
    
    
    def to_vector(sortedkeys):
        pass
    
    for p in posts_list_tokenized_stop_removed:
        post_corpus_counter+=  Counter(p)
        word_set = set()
        for word in p:
            word_set.add(word)
        for word in word_set:
            doc_req_counter[word] += 1


elif gram == 2:
# digram
    raise(NotImplementedError("work in progress"))

    for p in posts_list_tokenized[:10]:
        
        for i_digram in range(len(p)-1):
            # simple rules, both not stop words
            digram = tuple(p[i_digram:i_digram+2])
            if len(filter_word(digram)) > 0:
                post_corpus.add(digram)
                post_corpus_counter[digram]+=1
                pass
            
            pass



term_freq_dict = {k: v for (k, v) in post_corpus_counter.items()}







#%% sort keys according to alphabetical order
sorted_keys = sorted(term_freq_dict.keys())
import numpy as np
term_freq_vector_np = np.zeros([1, len(sorted_keys)])
for i, k in enumerate(sorted_keys):
    v = term_freq_dict[k]
    term_freq_vector_np[0][i] = v

#%% calculat IDF   http://www.tfidf.com/
idf_np = np.zeros([1, len(sorted_keys)])
n_doc = len(posts_list_tokenized_stop_removed)
for i, key in enumerate(sorted_keys):
    cnt = doc_req_counter[key]
    idf_np[0,i] = np.log(n_doc / cnt)  
    # suppose any high-frequency suppressing function work




#%%
l0  = len(posts_list)
l1 = len(posts_list_tokenized)
dl = l0-l1
print(f"before {len(posts_list)}, after {len(posts_list_tokenized)}, removed {dl}")

# helper function to vectorize any post vector given any global corpus
word_2_index = {}
for i , k in enumerate(sorted_keys):
    word_2_index[k] = i

def get_idf(word):
    if word not in word_2_index:
        return -1
    
    return idf_np[0, word_2_index[word]]

tf_idf_np = term_freq_vector_np * idf_np

def to_vector(post, sortedkeys = None):
    from collections import Counter
    
    sglobal = set(sortedkeys)
    spost = set(post)
    if not sglobal >= spost:
        raise Exception("post contains words not in global, not handled yet")
    post_vector_np = np.zeros([1, len(sorted_keys)])
    for word, cnt in dict(Counter(post)).items():
        idx = word_2_index[word]
        post_vector_np[0,idx] = cnt
    return post_vector_np

vectorized_post = []

for i, p in enumerate(posts_list_tokenized_stop_removed[:]):
    if i % 100 == 0:
        print(f"Done {i} posts")
    vectorized_post.append(to_vector(p, sortedkeys = sorted_keys) *
                           tf_idf_np)

#%% brute force search

# high dimensional nearest neighbour search brute force first

search_word_list = ["knowledge", "management"]
# uncomment below and pick a post to see if can find itself as a sanity test

# search_word_list = posts_list_tokenized_stop_removed[895]
search_filtered = filter_word(search_word_list)
vectorized_search = to_vector(search_filtered, sortedkeys = sorted_keys)

if len(search_filtered) > 0:
    # simplest case find 1 post only
    max_post = None
    max_similarity = 0
    for post_ID , tgt in enumerate(vectorized_post):
        siamese_cosine = np.sum(vectorized_search * tgt, axis = 1)
        siamese_cosine /= (np.sum(vectorized_search, axis = 1) * 
                           np.sum(tgt, axis = 1) )
        print(siamese_cosine)
        if siamese_cosine > max_similarity:
            max_similarity, max_post, max_post_ID = siamese_cosine, tgt, post_ID
        print(posts_list[max_post_ID])
    
else:
    raise ValueError("search keyworkds contain stopwords only")
#%%
# brute force done, now do some clustering on topics
""" many to choose from before needing to invent own:
    https://scikit-learn.org/stable/modules/clustering.html
    """

from sklearn.cluster import AgglomerativeClustering, Birch, DBSCAN, KMeans

vectorized_post2 = [v[0,:] for v in vectorized_post]

myCluster4 = KMeans(n_clusters = 10).fit(vectorized_post2[:1200])
myCluster3 = DBSCAN(n_clusters = 10).fit(vectorized_post2[:1200])
myCluster2 = Birch(n_clusters = 10).fit(vectorized_post2[:1200])
myCluster = AgglomerativeClustering(n_clusters = 10).fit(vectorized_post2[:1200])
from matplotlib import pyplot as plt
plt.plot(myCluster.labels_)



###

corpus = [" ".join(i) for i in posts_list_tokenized_stop_removed]
from sklearn import feature_extraction  
from sklearn.feature_extraction.text import TfidfTransformer  
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
transformer = TfidfTransformer()
tfidf = transformer.fit_transform(vectorizer.fit_transform(corpus))
word = vectorizer.get_feature_names()

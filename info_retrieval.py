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
def filter_word(list_token):
    return [word.lower() for word in list_token if word.isalpha() and  word not in stopwords_list]
    

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
            
    
    
    
    for p in posts_list_tokenized_stop_removed:
        post_corpus_counter+=  Counter(p)

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



post_freq_inv_vector_dict = {k: 1/v for (k, v) in post_corpus_counter.items()}







#%% sort keys according to alphabetical order
sorted_keys = sorted(post_freq_inv_vector_dict.keys())
import numpy as np
post_freq_inv_vector_np = np.zeros([1, len(sorted_keys)])
for i, k in enumerate(sorted_keys):
    v = post_freq_inv_vector_dict[k]
    post_freq_inv_vector_np[0][i] = v



#%%
l0  = len(posts_list)
l1 = len(posts_list_tokenized)
dl = l0-l1
print(f"before {len(posts_list)}, after {len(posts_list_tokenized)}, removed {dl}")

# helper function to vectorize any post vector given any global corpus
word_2_index = {}
for i , k in enumerate(sorted_keys):
    word_2_index[k] = i
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

for p in posts_list_tokenized_stop_removed[:]:
    vectorized_post.append(to_vector(p, sortedkeys = sorted_keys) *
                           post_freq_inv_vector_np)



# high dimensional nearest neighbour search brute force first

search_word_list = ["knowledge", "management"]
# uncomment and pick a post to see if can find itself
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

# brute force done, now do some clustering on topics
""" many to choose from before needing to invent own:
    https://scikit-learn.org/stable/modules/clustering.html
    """








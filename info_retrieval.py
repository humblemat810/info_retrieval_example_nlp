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
            to_append_content = to_append['content']
            posts_list.append(to_append_content)

                   
                   
import nltk
from nltk.corpus import stopwords




def is_valid_posts(p):
    if len(nltk.tokenize(p)) < 10 :
        return False
    
    return True

stopwords_list =  stopwords.words("english")

posts_list_tokenized = [nltk.tokenize.word_tokenize(p) for p in posts_list]

posts_list_tokenized_stop_removed = []
for p in posts_list_tokenized:
    p2 = [word.lower() for word in p if word.isalpha() and  word not in stopwords_list]
    posts_list_tokenized_stop_removed.append(p2 )
   

#%%    
# simplex case, monogram
def to_vector(sortedkeys):
    pass
post_corpus=set()
from collections import Counter
post_corpus_counter = Counter()
for p in posts_list_tokenized_stop_removed:
    post_corpus_counter+=  Counter(p)
        
post_freq_inv_vector_dict = {k: 1/v for (k, v) in post_corpus_counter.items()}

#%% sort keys according to alphabetical order
sorted_keys = sorted(post_freq_inv_vector_dict.keys())
import numpy as np
post_freq_inv_vector_np = np.zeros([1, len(sorted_keys)])
for i, k in enumerate(sorted_keys):
    v = post_freq_inv_vector_dict[k]
    post_freq_inv_vector_np[0][i] = v


from matplotlib import pyplot as plt
plt.plot(post_freq_inv_vector_np.transpose())

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

# test with 10 posts first
vectorized_post = []
for p in posts_list_tokenized_stop_removed[:10]:
    vectorized_post.append(to_vector(p, sortedkeys = sorted_keys))






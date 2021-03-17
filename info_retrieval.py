# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 23:17:42 2021

@author: ASUS
"""
import json
with open("sikm_groupsio-l.json", 'rb') as f:
    bstream = f.read()
txt = bstream.decode("utf-8")

txt_proper = txt.split("\n")

min_length_content = 10;

# fix json non standard format https://www.json.org/json-en.html
posts_list = []
for i, ln in enumerate(txt_proper):
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

post_corpus=set()
from collections import Counter
post_corpus_counter = Counter()
for p in posts_list_tokenized_stop_removed:
    post_corpus_counter+=  Counter(p)
        
post_freq_inv_vector = {k: 1/v for (k, v) in post_corpus_counter.items()}



#%%

print(f"before {len(posts_list)}, after {len(posts_list_tokenized)}")

for p in posts_list_tokenized:
    
    pass
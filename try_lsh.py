# check whether error in ipynb also occurs in pure python 


from load_coreferences import load_coreferences
import lsh 
import copy

import faiss 
import numpy as np

raw_mentions = load_coreferences(drop_duplicates=False)
mentions = {i: m for i, m in enumerate(raw_mentions)}
# stack them on top of each other 

mentions_scaled = copy.copy(mentions)

idx = len(mentions_scaled)
scaling_factor = 5
for i in range(1, scaling_factor):
    for idx_old in mentions.keys():
        m = mentions[idx_old]
        mentions_scaled[idx] = m 
        idx += 1


mylsh = lsh.LSHMinHash(mentions=mentions_scaled, shingle_size=4, signature_size=200, band_length=2)

mylsh.cluster()
mylsh.summarise()

mylsh.cluster(numpy_signature=True)
mylsh.summarise() # also gets killed 
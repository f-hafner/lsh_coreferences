"A quick sript to test REL hashing"


from load_coreferences import load_coreferences, load_pairs
from REL import lsh 
import numpy as np
import time 
from scipy import sparse 
import logging 
logging.basicConfig(level=logging.DEBUG)
import timeit 


raw_mentions = load_coreferences()
mentions = {i: m for i, m in enumerate(raw_mentions)}


mylsh = lsh.LSHMinHash(mentions=mentions, shingle_size=4, signature_size=50, band_length=2, sparse_binary=True)
mylsh.cluster()
mylsh.summarise()
mylsh_nobin = lsh.LSHMinHash(mentions=mentions, shingle_size=4, signature_size=50, band_length=2, sparse_binary=False)
mylsh_nobin.cluster()
mylsh_nobin.summarise()



# mylsh._build_vocab()
# mylsh.encode_binary(dest="sparse")

# mylsh.make_signature()


# mylsh.cluster()

# mylsh.candidates

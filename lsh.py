
from random import shuffle, seed 
import time 
import numpy as np 
import faiss
seed(3)

def split_mention(m):
    return m.split(" ")

def k_shingle(s, k):
    "convert string s into shingles of length k"
    shingle = []
    for i in range(len(s) - k + 1):
        shingle.append(s[i:(i+k)])
    return shingle


def partition_signature(s, b):
    "Convert signature s into b partitions of equal size"
    assert len(s) % b == 0
    rg = int(len(s) / b)
    partitions = []
    for i in range(0, len(s), rg):
        v = s[i:i+rg]
        partitions.append(v)
    return partitions

class LSHBase:
    def __init__(self, mentions, shingle_size):
        self.mentions = {i: {"shingles": k_shingle(m, shingle_size)} for i, m in mentions.items()}

    def _build_vocab(self):
        shingles = [v["shingles"] for v in self.mentions.values()]
        vocab = list(set([shingle for sublist in shingles for shingle in sublist]))
        self.vocab = vocab

    def encode_binary(self):
        for mention, data in self.mentions.items():
            v = [1 if self.vocab[i] in data["shingles"] else 0 for i in range(len(self.vocab)) ]
            self.mentions[mention] = {"shingles": data["shingles"] , "vector": v}




class LSHBitSampling(LSHBase):

    def __init__(self, mentions, shingle_size):
        super().__init__(mentions, shingle_size)

    def _extract_neighbors(self):
        neighbors = {}
        for i, k in enumerate(self.mentions.values()):
            n_idx = list(I[i])[1:] # ignore own
            n_i = [self.mentions[i] for i in n_idx]
            neighbors[k] = n_i
        return neighbors

    def _to_numpy(self):
        vectors = [v["vector"] for v in self.mentions.values()] 
        xb = np.stack([np.array(v) for v in vectors ]).astype('float32')
        self.xb = xb 

    def faiss_neighbors(self, k, nbits):
        start = time.time()
        self._build_vocab()
        self.encode_binary()
        self._to_numpy()
        d = self.xb.shape[1]
        # nbits = 100 # is this the length of the signature?? but the signature is already in the dense vector? 
        index = faiss.IndexLSH(d, nbits)   # build the index
        assert index.is_trained
        index.add(self.xb)                 # add vectors to the index

        # want k neighbors, but the first will be the vector itself. thus use k+1
        D, I = index.search(self.xb, k+1) # sanity check -- for short nbits, it may not even assign itself as the first closest neighbor
        end = time.time()
        self.D = D 
        self.I = I 
        self.timing = end - start

    def neighbors_to_dict(self, mention_dict):
        neighbors = {}
        start = time.time()
        for i in self.mentions.keys():
            n_idx = list(self.I[i])[1:]
            n_i = [mention_dict[i] for i in n_idx]
            k = mention_dict[i]
            neighbors[k] = n_i
        end = time.time()
        self.timing += start - end
        return neighbors
        
    def summarise(self):
        print(f"Took {self.timing} seconds to classify {len(self.mentions.keys())} mentions")


class LSHMinHash(LSHBase):

    def __init__(self, mentions, shingle_size, signature_size, n_buckets):
        super().__init__(mentions, shingle_size)
        if signature_size % n_buckets != 0:
            raise ValueError("Signature needs to be divisible into equal-sized buckets.")
        self.signature_size = signature_size
        self.n_buckets = n_buckets


    def _min_hash(self):
        signatures = {k: [] for k in self.mentions.keys()}
        hash = list(range(len(self.vocab)))

        for i in range(self.signature_size):
            shuffle(hash)
            # print(f"hash0: {hash0}")
            for k, v in self.mentions.items():
                vector = v["vector"]
                # print(f"{k}: {vector}")
                for i in range(len(hash)):
                    pos = hash[i]
                    # print(f"hash0[i] for {i} in {vector} is {vector[pos]}")
                    if vector[pos] == 1:
                        sig = pos # avoid trailing 0s
                        signatures[k].append(sig)
                        # print(f"signature for {k} is {signature}")
                        break 

        for k in self.mentions.keys():
            self.mentions[k]["signature"] = signatures[k]

    
    def _make_bands(self):
        for k, v in self.mentions.items():
            signature_bands = partition_signature(v["signature"], self.signature_size / self.n_buckets)
            self.mentions[k]["bands"] = signature_bands

    def _make_clusters(self):
        clusters = {m: [] for m in self.mentions.keys()}
        for m0, v1 in self.mentions.items():
            band0 = v1["bands"]
            for m1, v1 in self.mentions.items():
                if m0 != m1:
                    band1 = v1["bands"]
                    # if m0 == "albright" and m1 == "madeleine albright":
                    #     print("albright is here.")
                    #     print(f"band0: {band0}, band1: {band1}")
                    #     print(len(band1))
                    for rows0, rows1 in zip(band0, band1):
                        if rows0 == rows1:
                            # print(f"identified pair: {m0} and {m1}")
                            clusters[m0].append(m1)
                            break

        self.clusters = clusters

    
    def cluster(self):
        "Classify mentions into comparison groups"
        start = time.time()
        self._build_vocab()
        self.encode_binary()
        self._min_hash()
        self._make_bands()
        self._make_clusters()
        self.time = time.time() - start

    
    def summarise(self):
        sizes = [len(g) for g in self.clusters.values()]
        print(f"took {self.time} seconds for {len(self.clusters.keys())} mentions")
        print(f"average, min, max cluster size: {round(sum(sizes)/len(sizes),2)}, {min(sizes)}, {max(sizes)}")

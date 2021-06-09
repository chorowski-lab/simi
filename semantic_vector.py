from scipy.stats.mstats import spearmanr
import numpy as np
import matplotlib.pyplot as plt
import sys
import editdistance
import random
from gensim.models import Word2Vec
from corpora_simplification import quantized_sequence_compress

class SemanticVectors:
    def __init__(self, model_name):
        self.model_words = Word2Vec.load(model_name)
        self.words_list = list(self.model_words.wv.vocab)

        self.cache = {}

    def get_semantic_vector(self, seq):
        key = ','.join(seq)
        
        if key in self.cache:
            return self.cache[key]
                
        w = quantized_sequence_compress(seq)
        
        #w = w.lower()   

        ds = [ (editdistance.eval(w, w0),w0) for w0 in self.words_list]
        m, _ = min(ds)
        
        candidates = [w for (d0, w) in ds if d0 == m]
        
        v = np.zeros(100)
        
        for c in candidates:
            v += self.model_words.wv[c]
        
        v /= len(candidates)
        self.cache[key] = v    
        return v 
    


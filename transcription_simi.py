# %% IMPORTS & DECLARATIONS
import os
from progressbar import ProgressBar
import random
import numpy as np
import sentencepiece
from sklearn.cluster import KMeans
import pathlib
import pandas
import sys
from itertools import product
from src import *
from env import *

# %% MAIN PROCESS DEFINITION

def run(config, data):
    vocab_size = config['sp_vocab_size_1']
    seed = config['seed']

    SP_PREFIX_PATH = f'{ROOTPATH}/models/transcriptions/sentencepiece/{data.name}/vocab_size_{vocab_size}/s{seed}'
    sp_formatted, sp_segmentation = segment(data.data, SP_PREFIX_PATH, vocab_size)
    
    word2vec_size = config['word2vec_size']
    W2V_PATH = f'{ROOTPATH}/models/transcriptions/word2vec/{data.name}/vocab_size_{vocab_size}/w2v_size_{word2vec_size}/s{seed}.model'
    w2v_encodings, w2v_weights, reconstruct = vectorize(sp_formatted, W2V_PATH, word2vec_size)

    n_clusters = config['kmeans_n_clusters']
    KMEANS_PREFIX_PATH = f'./models/transcriptions/kmeans/{data.name}/vocab_size_{vocab_size}/w2v_size_{word2vec_size}/n_clusters_{n_clusters}/s{seed}'
    kmeans_formatted = reconstruct(cluster_kmeans(w2v_encodings, w2v_weights, KMEANS_PREFIX_PATH, n_clusters))

    reformatted = [int_array_to_string(sentence) for sentence in kmeans_formatted]

    vocab_size_2 = config['sp_vocab_size_2']
    SP2_PREFIX_PATH = f'./models/transcriptions/sentencepiece_after/{data.name}/vocab_size_{vocab_size}/w2v_size_{word2vec_size}/n_clusters_{n_clusters}/after_vocab_size_{vocab_size_2}/s{seed}'
    SP2_MODEL_PATH = SP2_PREFIX_PATH + '.model'

    sp2_formatted, sp2_segmentation_internal = segment(kmeans_formatted, SP2_PREFIX_PATH, vocab_size_2)

    sp2_segmentation = merge_segmentations(sp_segmentation, sp2_segmentation_internal)

    r1 = data.f1score(sp_segmentation)
    save_results(config, r1, 'iter1')
    r2 = data.f1score(sp2_segmentation)
    save_results(config, r2, 'iter2')
    
if False:
    config = {
        'seed': 290956,
        'dataset': 'm2/dev-clean',
        'sp_vocab_size_1': 10000,
        'sp_vocab_size_2': 10000,
        'word2vec_size': 100,
        'kmeans_n_clusters': 100
    }
    print(config, flush=True)

    seed = config['seed']
    random.seed(seed)
    np.random.seed(seed)
    sentencepiece.set_random_generator_seed(seed)

    data = Dataset(config['dataset'])
    run(config, data)
    sys.exit(0)

# %% RUN

for dataset, vs1, vs2 in product(
    ['m1/train-full-960', 'm2/train-full-960', 'm3/train-full-960'],
    [200, 500, 1000], # [2000, 5000, 10000, 20000, 40000, 80000, 150000, 300000, 500000, 1000000],
    [2000, 5000, 10000, 20000, 40000, 80000, 150000, 300000, 500000, 1000000]):
    config = {
        'seed': 290956,
        'dataset': dataset,
        'sp_vocab_size_1': vs1,
        'sp_vocab_size_2': vs2,
        'word2vec_size': 100,
        'kmeans_n_clusters': 100
    }
    print(config, flush=True)

    seed = config['seed']
    random.seed(seed)
    np.random.seed(seed)
    sentencepiece.set_random_generator_seed(seed)

    data = Dataset(config['dataset'])
    try:
        run(config, data)
    except KeyboardInterrupt:
        raise KeyboardInterrupt
    except:
        pass

sys.exit(0)

# %% TRAIN SENTENCEPIECE
vocab_size = 3000
SentencePieceTrainer.train(sentence_iterator=iter(data.data), model_prefix=f'./models/transcriptions/sentencepiece/{dataset}_{vocab_size}', vocab_size=vocab_size)

# %% ENCODE WITH SENTENCEPIECE
sp = SentencePieceProcessor()
sp.Load(f'./models/transcriptions/sentencepiece/{dataset}_{vocab_size}.model')
encodings = sp.Encode(data.data, out_type=str)
formatted = list(' '.join(encoding).replace('▁', '').strip() for encoding in encodings)

# %% SCORE SENTENCEPIECE ENCODINGS
data.f1score(formatted)
print(formatted[0])
print(formatted[1])
print(formatted[182])

# %% TRAIN WORD2VEC

model = Word2Vec(sentences=encodings)
model.save(f'./models/transcriptions/word2vec/{dataset}.model')

# %% LOAD WORD2VEC

model = Word2Vec.load(f'./models/transcriptions/word2vec/{dataset}_{vocab_size}.model')

# %% TEST WORD2VEC

print(model.wv.most_similar('england'))

# %% RUN WORD2VEC ON GROUD-TRUTH

sentences = list(map(lambda line: line.split(), data.gt))
vocab_size = 1000

model = Word2Vec(sentences=sentences)

# model.save(f'./models/transcriptions/word2vec/{dataset}_{vocab_size}_GT_TEST.model')
model.wv.most_similar('hollow')
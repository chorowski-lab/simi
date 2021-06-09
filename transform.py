# %% IMPORTS & DECLARATIONS
import os
from progressbar import ProgressBar
import random
import numpy as np
from sklearn.cluster import KMeans
import pathlib
import pandas
import sys
from itertools import product
from src import *
from env import *
import yaml

# %%

def run(config):
    trainset = config["trainset"]
    evalset = config["evalset"] if "evalset" in config.keys() else config["trainset"]
    seed = config["seed"]
    vocab_size = config["vocab_size"]
    w2v_size = config["word2vec_size"]
    n_clusters = config["kmeans_n_clusters"]

    data = Data(f'/pio/data/zerospeech2021/quantized/{evalset}/quantized_outputs.txt')

    train = evalset == trainset

    random.seed(seed)
    np.random.seed(seed)
    sentencepiece.set_random_generator_seed(seed)
    
    sp_prefix_path = f'{ROOTPATH}/models/sentencepiece/{trainset}/vocab_size_{vocab_size}/s{seed}'
    w2v_path = f'{ROOTPATH}/models/word2vec/{trainset}/vocab_size_{vocab_size}/w2v_size_{w2v_size}/s{seed}'
    kmeans_path = f'{ROOTPATH}/models/kmeans/{trainset}/vocab_size_{vocab_size}/w2v_size_{w2v_size}/n_clusters_{n_clusters}/s{seed}'

    sp_formatted, sp_segmentation = segment(data.data, sp_prefix_path, vocab_size, train)
    w2v_encodings, w2v_weights, reconstruct = vectorize(sp_formatted, w2v_path, w2v_size, train)
    labels = cluster_kmeans(w2v_encodings, w2v_weights, kmeans_path, n_clusters, train)
    formatted_output = reconstruct(labels)

    if 'output_result' in config.keys():
        configpath = os.path.join(config['output_result'], 'config.yaml')
        outputpath = os.path.join(config['output_result'], 'quantized_outputs.txt')
        ensure_path(configpath)
        with open(configpath, 'wb') as cfg_file:
            cfg_file.write(yaml.dump(config, encoding='utf-8'))

        with open(outputpath, 'w', encoding='utf8') as output:
            for i in range(len(formatted_output)):
                a = formatted_output[i]
                b = ",".join(map(str, a))
                output.write(f'{data.filenames[i]} {b}\n')

# %%

trainset = 'LibriSpeech/train-full-960'

base_output = '/pio/scratch/1/i290956/zs2021/simi/results/train-full-960_vs500_w2v100_ncl50/'
config = {
    'seed': 290956,
    'evalset': trainset,
    'trainset': trainset,
    'vocab_size': 500,
    'word2vec_size': 100,
    'kmeans_n_clusters': 50,
    'output_result': base_output
}


run(config)

for evalset in ['semantic/dev/librispeech', 'semantic/dev/synthetic']:
    config['evalset'] = evalset
    config['output_result'] = base_output + evalset
    run(config)

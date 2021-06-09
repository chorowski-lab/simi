# %% IMPORTS & DECLARATIONS
import random

import numpy as np
import sentencepiece
from more_itertools import grouper
from argparse import ArgumentParser
from env import *
from src import *

# %%

def parseArgs():
    parser = ArgumentParser()

    parser.add_argument('pipeline', type=str, help='Pipeline description in format <n>-<m>-<n>-...')
    parser.add_argument('--viterbi', action='store_true')

    return parser.parse_args()


def run(config, args):
    seed = config["seed"]
    w2v_size = config["word2vec_size"]
    dataset = config["dataset"]
    squash = "squash" in config.keys() and config["squash"]
    random.seed(seed)
    np.random.seed(seed)
    sentencepiece.set_random_generator_seed(seed)


    data = LibriSpeech(dataset)

    if squash:
        segmentation = data.squash()

    initial_vocab_size = config["pipeline"][0]
    foldername = f'{dataset}/w2v{w2v_size}{"_squashed" if squash else ""}{"_viterbi" if args.viterbi else ""}'
    pipeline = str(initial_vocab_size)
    # initial segmentation
    
    sp_prefix_path = f'{ROOTPATH}/models/sentencepiece/{foldername}/{pipeline}/s{seed}'
    sp_data = train_sentencepiece(data.data, sp_prefix_path, initial_vocab_size)
    if args.viterbi:
        sp_formatted, sp_segmentation = segment_viterbi(sp_data, data.clusterings, sp_prefix_path)
    else:
        sp_formatted, sp_segmentation = segment_sentencepiece(sp_data, sp_prefix_path)

    if squash:
        sp_segmentation = merge_segmentations(segmentation, sp_segmentation)

    words_score, phones_score = data.rate_segmentation(sp_segmentation)
    save_pipeline_results(config, pipeline, words_score, phones_score, args.viterbi)

    for n_clusters, vocab_size in grouper(config["pipeline"][1:], 2):

        w2v_path = f'{ROOTPATH}/models/word2vec/{foldername}/{pipeline}/s{seed}'  
        w2v_encodings, w2v_weights, reconstruct = vectorize(sp_formatted, w2v_path, w2v_size)

        pipeline += '-' + str(n_clusters)
        kmeans_path = f'{ROOTPATH}/models/kmeans/{foldername}/{pipeline}/s{seed}'
        labels = cluster_kmeans(w2v_encodings, w2v_weights, kmeans_path, n_clusters)
        
        pipeline += '-' + str(vocab_size)
        sp_prefix_path = f'{ROOTPATH}/models/sentencepiece/{foldername}/{pipeline}/s{seed}'
        sp_data = train_sentencepiece(reconstruct(labels), sp_prefix_path, vocab_size)
        sp_formatted, sp2_segmentation_internal = segment_sentencepiece(sp_data, sp_prefix_path)
        
        sp_segmentation = merge_segmentations(sp_segmentation, sp2_segmentation_internal)

        words_score, phones_score = data.rate_segmentation(sp_segmentation)
        save_pipeline_results(config, pipeline, words_score, phones_score, args.viterbi)
   
# %%


config = {
    'seed': 290956,
    'dataset': 'LibriSpeech/train-clean-100',
    'word2vec_size': 100,
    'squash': True,
    'pipeline': [
        200,
        50,
        50000
    ]
}

# if __name__ == "__main__":
#     args = parseArgs()
#     config['pipeline'] = format_pipeline(args.pipeline)
#     for squash in [True, False]:
#         config['squash'] = squash
#         try:
#             run(config)
#         except:
#             pass

args = parseArgs()
for squash in [False, True]:
    for p1 in [200, 500, 1000, 2000, 5000, 10000, 20000, 50000]:
        for p2 in [200]:
            config['pipeline'] = format_pipeline(f'{p1}')
            config['squash'] = squash
            # run(config)
            try:
                run(config, args)
            except Exception as e:
                print(e)

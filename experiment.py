import os
import pathlib
import random
import sys
from argparse import ArgumentParser

import numpy as np
import sentencepiece
from scipy.special import logsumexp
from collections import defaultdict

from simi import dataset, utils
from simi.segmentation import segment_sentencepiece, segment_viterbi
from simi.dataset import Data

from fst_sp.utils import SentencePiece
from fst_sp.spt_trainer import SentencePieceTrainer

sys.setrecursionlimit(10**6)

def parseArgs():
    parser = ArgumentParser()

    parser.add_argument('word_map', type=pathlib.Path,
                        help='Path to the word map')
    parser.add_argument('sentencepiece_prefix', type=pathlib.Path,
                        help='Prefix for sentencepiece model. It must point to existing folder containing sentencepiece model & vocab.')
    parser.add_argument('trainset', type=pathlib.Path,
                        help='Path to the quantized trainset for sentencepiece pruning.')
    parser.add_argument('output', type=pathlib.Path,
                        help='Output path')
    parser.add_argument('--vocab_size', type=int, default=10000,
                        help='Vocabulary size')
    parser.add_argument('--seed', type=int, default=290956,
                        help='Random seed')

    return parser.parse_args()


def run(args):
    word_to_id = {l.split()[0]: l.split()[1] for l in open(args.word_map, 'r', encoding='utf8')}
    word_to_prob = {l.split()[0].replace('▁', ''): float(l.split()[1]) for l in open(args.sentencepiece_prefix.with_suffix('.vocab'), 'r', encoding='utf8')}
    id_to_words = defaultdict(list)
    for word, id in word_to_id.items():
        id_to_words[id].append(word)

    pieces = []
    letters = set()
    k = 3
    for id, words in id_to_words.items():
        log_probs = [word_to_prob[w] for w in words]
        weights = np.exp(log_probs) / np.sum(np.exp(log_probs))
        pieces.append(SentencePiece(int(id)+2, '|'.join(words), logsumexp(log_probs), weights))
        k = max(k, int(id)+3)
        for w in words:
            if len(w) == 1:
                letters.add(w)

    # for word, prob in word_to_prob.items():
    #     if len(w) == 1 and w not in letters:
    #         words.add(w)
    #         pieces.append(SentencePiece(k, w, prob))
    #         k += 1
    
    for w in letters:
        pieces.append(SentencePiece(k, w, word_to_prob[w]))
        k += 1

    trainset = Data(args.trainset)
    model = SentencePieceTrainer.train(utils.data_to_string_arrays(trainset.data), [SentencePiece(index=0, symbol='<unk>', log_freq=0)] + pieces, vocab_size=args.vocab_size, complex_pieces=True)
    model.save(args.output)
    print(model.pieces)


if __name__ == "__main__":
    args = parseArgs()
    run(args)
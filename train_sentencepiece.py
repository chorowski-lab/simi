import os
import pathlib
import random
from argparse import ArgumentParser

import numpy as np
import sentencepiece

from simi.utils import ensure_path
from simi.dataset import Data
from simi.segmentation import train_sentencepiece


def parseArgs():
    parser = ArgumentParser()

    parser.add_argument('trainset', type=pathlib.Path,
                        help='Path to the quantized trainset for sentencepiece learning.')
    parser.add_argument('sentencepiece_prefix', type=pathlib.Path,
                        help='Prefix for the sentencepiece model, last item in path should be the model\'s name')
    parser.add_argument('vocab_size', type=int,
                        help='Sentencepiece\'s vocabulary size.')
    parser.add_argument('--seed', type=int, default=290956,
                        help='Random seed')
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite existing sentencepiece model')
    parser.add_argument('--max_piece_length', type=int, default=100,
                        help='Max length of sentence piece. Default=100')
    return parser.parse_args()


def run(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    sentencepiece.set_random_generator_seed(args.seed)
    
    assert not ensure_path(args.sentencepiece_prefix.with_suffix('.model')) or args.overwrite, \
        f'Sentencepiece model found at {args.sentencepiece_prefix}. If you want to overwrite, rerun with --overwrite.'
        
    print('Loading trainset...')
    trainset = Data(args.trainset)
    
    train_sentencepiece(trainset.data, args.sentencepiece_prefix, args.vocab_size, max_piece_length=args.max_piece_length)

    
if __name__ == "__main__":
    args = parseArgs()
    run(args)

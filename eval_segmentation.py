import os
import pathlib
import random
from argparse import ArgumentError, ArgumentParser
import pickle

import numpy as np
import sentencepiece

from simi import utils, dataset
from simi.segmentation import segment_sentencepiece, segment_viterbi, train_sentencepiece


def parseArgs():
    parser = ArgumentParser()

    parser.add_argument('dataset', type=pathlib.Path,
                        help='Path to the quantized dataset, which is to be segmented')
    parser.add_argument('sentencepiece_prefix', type=pathlib.Path,
                        help='Prefix for sentencepiece model. It must point to existing folder containing sentencepiece model & vocab.')
    parser.add_argument('output', type=pathlib.Path,
                        help='Output folder')
    parser.add_argument('--clusterings', type=str,
                        help='Path to the clusterings of the data, must match the dataset. Required if using Viterbi segmentation')
    parser.add_argument('--seed', type=int, default=290956,
                        help='Random seed')
    parser.add_argument('--viterbi', action='store_true',
                        help='Do Viterbi segmentation instead of sentencepiece\'s default')
    parser.add_argument('--alpha', type=float, default=1.0, 
                        help='Temperature for sharpening/smoothening clustering distribution. More than 1: sharpening, less than 1: smoothening. Deafult: 1.0')
    parser.add_argument('--output_format', type=str, default='txt,csv',
                        help='Output format of the transformed dataset. Comma separated list of: \'txt\' (arrays of strings) or \'csv\' (similar to LibriSpeech alignments)')

    return parser.parse_args()


def save_segmentation(formatted, dataset, path, args):
    if not args.output_format or not args.output_format.split(','):
        raise Exception(f'Output format should be a comma sparated list')

    
    if 'txt' in args.output_format.split(','):
        with open(args.output / 'segmented_outputs.txt', 'w') as output:
            for sentence, fname in zip(formatted, dataset.filenames):
                output.write(f'{fname} {" ".join(sentence)}\n')

    if 'csv' in args.output_format.split(','):
        if not os.path.exists(path):
            os.makedirs(path)
        for sentence, fname in zip(formatted, dataset.filenames):
            with open(path / (fname+'.csv'), 'w') as output:
                i = 0
                for word in sentence:
                    output.write(f'{i/100},{(i+len(word))/100},{word},phones\n')
                    i += len(word)


def run(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    sentencepiece.set_random_generator_seed(args.seed)
    
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    
    assert utils.ensure_path(f'{args.sentencepiece_prefix}.model'), \
        f'Sentencepiece model not found at {args.sentencepiece_prefix}'

    print('Loading dataset...')
    devset = dataset.Data(args.dataset)

    if args.viterbi:
        print('Running Viterbi segmentation...')
        assert args.clusterings is not None, "If viterbi is used you have to specify path to the clusterings"
        devset.load_clusterings(args.clusterings, args.alpha)

        vit_formatted, vit_segmentation = segment_viterbi(devset.data, devset.clusterings, args.sentencepiece_prefix)
        save_segmentation(vit_formatted, devset, args.output, args)

    else:
        print('Running SentencePiece segmentation...')
        sp_formatted, sp_segmentation = segment_sentencepiece(devset.data, args.sentencepiece_prefix)
        save_segmentation(sp_formatted, devset, args.output, args)
    

if __name__ == "__main__":
    args = parseArgs()
    run(args)

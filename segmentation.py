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

    parser.add_argument('output', type=pathlib.Path,
                        help='Output folder')
    parser.add_argument('--dataset', type=pathlib.Path,
                        help='Path to the quantized dataset, which is to be segmented')
    parser.add_argument('--trainset', type=pathlib.Path,
                        help='Path to the quantized trainset for sentencepiece learning. Necessary if --sentencepiece_prefix does not point to existing folder containing sentencepiece model & vocab.')
    parser.add_argument('--vocab_size', type=int,
                        help='Sentencepiece\'s vocabulary size. Necessary if --sentencepiece_prefix does not point to existing folder containing sentencepiece model & vocab.')
    parser.add_argument('--sentencepiece_prefix', type=pathlib.Path,
                        help='Prefix for sentencepiece model, defaults to output+\'sentencepiece\'. In eval mode must point to existing folder containing sentencepiece model & vocab.')
    parser.add_argument('--clusterings', type=str,
                        help='Path to the clusterings of the data, must match the dataset. Required if using Viterbi segmentation')
    parser.add_argument('--seed', type=int, default=290956,
                        help='Random seed')
    parser.add_argument('--viterbi', action='store_true',
                        help='Do Viterbi segmentation instead of sentencepiece\'s default')
    parser.add_argument('--eval', action='store_true',
                        help='Run in eval only mode (do not train sentencepiece). If set, then arguments realted to sentencepiece training are ignored')
    parser.add_argument('--train', action='store_true',
                        help='Run in train only mode (do not segment dataset). If set, then arguments related to segmentation are ignored')
    parser.add_argument('--alpha', type=float, default=1.0, 
                        help='Temperature for sharpening/smoothening clustering distribution. More than 1: sharpening, less than 1: smoothening. Deafult: 1.0')
    parser.add_argument('--data_output_format', type=str, default='str',
                        help='Output format of the transformed dataset. Either \'str\' (arrays of strings, deafult) or \'pkl\' (similar to the baseline quantization, as pickle)')
    parser.add_argument('--segmentation_output_format', type=str, default='csv',
                        help='Output format of the segmentation. Either \'pkl\' (pickled array of breakpoints) or \'csv\' (similar to LibriSpeech alignments, default)')
    parser.add_argument('--max_piece_length', type=int, default=100,
                        help='Max length of sentence piece. Default=100')
    return parser.parse_args()


def save_data(formatted, dataset, args):
    if args.data_output_format == 'str':
        with open(args.output / 'segmented_outputs.txt', 'w') as output:
            for sentence, fname in zip(formatted, dataset.filenames):
                output.write(f'{fname} {" ".join(sentence)}\n')

    elif args.data_output_format == 'pkl':
        with open(args.output / 'segmented_outputs.pkl', 'wb') as output:
            pickle.dump(formatted, output)

    else:
        raise ArgumentError(f'Invalid data_output_format, should be \'str\' or \'pkl\', but got \'{args.data_output_format}\'')


def save_segmentation(segmentation, formatted, dataset, path, args):
    if args.segmentation_output_format == 'pkl':
        with open(str(path) + '.pkl', 'wb') as output:
            pickle.dump(segmentation, output)

    elif args.segmentation_output_format == 'csv':
        if not os.path.exists(path):
            os.makedirs(path)
        for sentence, fname in zip(formatted, dataset.filenames):
            with open(path / (fname+'.csv'), 'w') as output:
                i = 0
                for word in sentence:
                    output.write(f'{i/100},{(i+len(word))/100},{word},phones\n')
                    i += len(word)

    else:
        raise ArgumentError(f'Invalid segmentation_output_format, should be \'pkl\' or \'csv\', but got \'{args.segmentation_output_format}\'')


def run(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    sentencepiece.set_random_generator_seed(args.seed)
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    
    if args.sentencepiece_prefix is None:
        sp_prefix_path = args.output / 'sentencepiece'
    else:
        sp_prefix_path = args.sentencepiece_prefix

    if not utils.ensure_path(sp_prefix_path.with_suffix('.model')):
        assert not args.eval, ""
        print('Loading trainset...')
        trainset = dataset.Data(args.trainset)
        train_sentencepiece(trainset.data, sp_prefix_path, args.vocab_size, max_piece_length=args.max_piece_length)

    print('Loading devset...')
    devset = dataset.Data(args.dataset)

    if args.viterbi:
        print('Running Viterbi segmentation...')
        assert args.clusterings is not None, "If viterbi is used you have to specify path to the clusterings"
        devset.load_clusterings(args.clusterings, args.alpha)

        vit_formatted, vit_segmentation = segment_viterbi(devset.data, devset.clusterings, sp_prefix_path)
        save_data(vit_formatted, devset, args)
        save_segmentation(vit_segmentation, vit_formatted, devset, args.output, args)

    else:
        print('Running SentencePiece segmentation...')
        sp_formatted, sp_segmentation = segment_sentencepiece(devset.data, sp_prefix_path)
        save_data(sp_formatted, devset, args)
        save_segmentation(sp_segmentation, sp_formatted, devset, args.output, args)
    

if __name__ == "__main__":
    args = parseArgs()
    run(args)

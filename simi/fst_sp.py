### Finite state transducer reimplementation of SentencePiece

# You need to source config.sh to omitt import problems (by setting PYTHONPATH)

# Example use:
#   python simi/fst_sp.py --trainset simi/fst_sp/botchan_small.txt  model

import os,sys
import simi
from simi.stringify import *
import pathlib
import random
from argparse import ArgumentError, ArgumentParser
import utils,dataset
import pickle

from collections import Counter
from fst_sp.esa import ESA


import fst_sp.kaldi_fst_sp as fst
import numpy as np

#import sentencepiece as spm

print_ = lambda x: None #do nothing if verbose not active

def parseArgs():
    parser = ArgumentParser()

    parser.add_argument('output', type=pathlib.Path,
                        help='Output folder')
    parser.add_argument('--delimiter', type=str, default="$",
                        help='Delimiter symbol')
    parser.add_argument('--max_piece_length', type=int, default=10,
                        help='maximum length of sentence piece. Default: 10')
    parser.add_argument('--sentencepiece_prefix', type=pathlib.Path,
                        help='Prefix for sentencepiece model, defaults to output+\'sentencepiece\'. In eval mode must point to existing folder containing sentencepiece model & vocab.')
    parser.add_argument('--trainset', type=pathlib.Path,
                        help='Path to the text for sentencepiece learning.')# Necessary if --sentencepiece_prefix does not point to existing folder containing sentencepiece model & vocab.'
    parser.add_argument('--seed_sentencepiece_size', type=int, default=1000000,
                        help='the size of seed sentencepieces')
    parser.add_argument('-v','--verbose', dest='verbose', action='store_true')
    parser.add_argument('--vocab_size', type=int, default = 8000,
                        help='Sentencepiece\'s vocabulary size. Default: 8000' )
    parser.add_argument('--shrinking_factor', type=float, default = 0.75,
                        help='Keeps top shrinking_factor pieces with respect to the loss. Default: 0.75')
    parser.add_argument('--num_sub_iterations', type=int, default = 2,
                        help='Number of EM sub-iterations. Default: 2')

            
    #parser.add_argument('--vocab_size', type=int,
    #                    help='Sentencepiece\'s vocabulary size. Necessary if --sentencepiece_prefix does not point to existing folder containing sentencepiece model & vocab.')
    #parser.add_argument('--dataset', type=pathlib.Path,
    #                    help='Path to the quantized dataset, which is to be segmented')
    #parser.add_argument('--clusterings', type=str,
    #                    help='Path to the clusterings of the data, must match the dataset. Required if using Viterbi segmentation')
    #parser.add_argument('--seed', type=int, default=290956,
    #                    help='Random seed')
    #parser.add_argument('--viterbi', action='store_true',
    #                    help='Do Viterbi segmentation instead of sentencepiece\'s default')
    #parser.add_argument('--eval', action='store_true',
    #                    help='Run in eval only mode (do not train sentencepiece). If set, then arguments realted to sentencepiece training are ignored')
    #parser.add_argument('--train', action='store_true',
    #                    help='Run in train only mode (do not segment dataset). If set, then arguments related to segmentation are ignored')
    #parser.add_argument('--alpha', type=float, default=1.0, 
    #                    help='Temperature for sharpening/smoothening clustering distribution. More than 1: sharpening, less than 1: smoothening. Deafult: 1.0')
    #parser.add_argument('--data_output_format', type=str, default='str',
    #                    help='Output format of the transformed dataset. Either \'str\' (arrays of strings, deafult) or \'pkl\' (similar to the baseline quantization, as pickle)')
    #parser.add_argument('--segmentation_output_format', type=str, default='csv',
    #                    help='Output format of the segmentation. Either \'pkl\' (pickled array of breakpoints) or \'csv\' (similar to LibriSpeech alignments, default)')
    #parser.add_argument('--max_piece_length', type=int, default=100,
    #                    help='Max length of sentence piece. Default=100')
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
    #random.seed(args.seed)
    #np.random.seed(args.seed)
    #sentencepiece.set_random_generator_seed(args.seed)
    #if not os.path.exists(args.output):
    #    os.makedirs(args.output)
    debug = args.verbose

    print_ = lambda x: None
    if args.verbose:
        print_ = print
    
    if args.sentencepiece_prefix is None:
        sp_prefix_path = args.output / 'sentencepiece'
    else:
        sp_prefix_path = args.sentencepiece_prefix

    if not utils.ensure_path(sp_prefix_path.with_suffix('.model')):
        #assert not args.eval, ""
        print('Loading trainset...')

        sentences = []
        for line in open(args.trainset, 'r', encoding='utf8'):
            sentences.append(line.strip()+args.delimiter) # XXX args.delimieter at the end of line?
        #trainset = dataset.Data(args.trainset)
        #train_sentencepiece(trainset.data, sp_prefix_path, args.vocab_size, max_piece_length=args.max_piece_length)

    seed_sp = make_seed_sentence_pieces(sentences,
                    seed_vocab_size = args.seed_sentencepiece_size, 
                    max_piece_length = args.max_piece_length,
                    debug=debug)
    #for a,b in seed_sp:
    #    print(a,b)


    #Sentencepiece training
    pieces = [fst.SentencePiece(ind,symb,log_freq) for ind,(symb,log_freq) in enumerate(seed_sp)]
    T=fst.SentencePieceTrainer(pieces)
    sentences = [fst.Sentence(text, 1) for text in sentences]

    DESIRED_PIECES = args.vocab_size
    PRUNE_FRAC = args.shrinking_factor
    NUM_SUBITER = args.num_sub_iterations

    while True:
        # EM Step
        for sub_iter in range(NUM_SUBITER):  
            e_ret = T.EStep(pieces, sentences)
            pieces = T.MStep(pieces, e_ret.counts)
            print(f"EM sub_iter={sub_iter} size={len(pieces)} tot_piece_prob={np.exp(logsumexp([piece.log_freq for piece in pieces]))} "
                f"obj={e_ret.objective} num_tokens={e_ret.n_tokens} num_tokens/piece={e_ret.n_tokens / len(pieces)}" )
        
        if len(pieces) <= DESIRED_PIECES:
            break

        pieces = T.prune_pieces(pieces, sentences, DESIRED_PIECES, PRUNE_FRAC)
        if len(pieces) <= DESIRED_PIECES:
            break
            
    print(pieces)
    # TODO: add finalization

    #print('Loading devset...')
    #devset = dataset.Data(args.dataset)

    #if args.viterbi:
    #    print('Running Viterbi segmentation...')
    #    assert args.clusterings is not None, "If viterbi is used you have to specify path to the clusterings"
    #    devset.load_clusterings(args.clusterings, args.alpha)

    #    vit_formatted, vit_segmentation = segment_viterbi(devset.data, devset.clusterings, sp_prefix_path)
    #    save_data(vit_formatted, devset, args)
    #    save_segmentation(vit_segmentation, vit_formatted, devset, args.output, args)

    #else:
    #    print('Running SentencePiece segmentation...')
    #    sp_formatted, sp_segmentation = segment_sentencepiece(devset.data, sp_prefix_path)
    #    save_data(sp_formatted, devset, args)
    #    save_segmentation(sp_segmentation, sp_formatted, devset, args.output, args)


def to_log_prob(pieces):
    Z = np.log(sum(score for p, score in pieces))
    pieces = [(p, np.log(score) - Z) for p, score in pieces]
    return pieces


def make_seed_sentence_pieces(sentences, seed_vocab_size, max_piece_length, debug=False,
                                delimiter='#'):
    print("Extracting frequent sub strings...")

    # Makes an enhanced suffix array to extract all sub strings occurring
    # more than 2 times in the sentence.
    esa = ESA()
    esa.fit(sentences, delimiter=delimiter, max_piece_len=max_piece_length, debug = debug)

    seed_sentp = sorted(esa.pieces(), key=lambda p_score: -p_score[1])

    # Prune
    seed_sentp = seed_sentp[:seed_vocab_size]

    # all_chars must be included in the seed sentencepieces.
    all_chars = Counter()
    for s in sentences:
        all_chars.update(s)
    del all_chars[delimiter]

    for c, cnt in all_chars.items():
        seed_sentp.append((c, cnt))  # 0.5)) # XXX XXX XXX

    seed_sentp = to_log_prob(seed_sentp)
    seed_sentp = sorted(esa.pieces(), key=lambda p_score: -p_score[1])

    print(f"Initialized {len(seed_sentp)} seed sentencepieces")

    return seed_sentp
    
if __name__=="__main__":
    args = parseArgs()
    run(args)

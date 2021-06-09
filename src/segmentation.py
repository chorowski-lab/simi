# %%
import os
from collections import defaultdict

import numpy as np
import tqdm
from progressbar import ProgressBar
from sentencepiece import SentencePieceProcessor, SentencePieceTrainer

from src.viterbi import sentpiece_viterbi

from .score import Score
from .utils import *

ALIGNMENTS_ROOTPATH = '/pio/data/zerospeech2021/librispeech_alignments'


def rate_segmentation(gt_segs, es_segs, tolerance=2):
    Nhit, Nref, Nf = 0, 0, 0
    for gt_seg, es_seg in zip(gt_segs, es_segs):
        Nref += len(gt_seg)
        Nf += len(es_seg)
        i, j = 0, 0
        while i < len(gt_seg) and j < len(es_seg):
            if gt_seg[i] <= es_seg[j] + tolerance and gt_seg[i] + tolerance >= es_seg[j]:
                Nhit += 1
                i += 1
                j += 1
            elif gt_seg[i] + tolerance < es_seg[j]:
                i += 1
            else:
                j += 1
    return Score(Nhit, Nref, Nf)


class LibriSpeechSegmentation(object):
    def __init__(self, name):
        self.phones = defaultdict(list)
        self.words = defaultdict(list)
        self.phone_list = set()
        for root, dirs, files in os.walk(os.path.join(ALIGNMENTS_ROOTPATH, name)):
            for file in files:
                if file.endswith('.csv'):
                    for line in open(os.path.join(root, file), 'r', encoding='utf8'):
                        t1, t2, q, kind = line.strip().split(',')
                        if kind == 'phones':
                            self.phone_list.add(q)
                        n1, n2 = int(float(t1)*100), int(float(t2)*100)
                        assert n1 < n2
                        d = self.words if kind == 'words' else self.phones
                        fname = file[:-4]
                        if fname in d.keys():
                            assert d[fname][-1] <= n1

                            if d[fname][-1] != n1:
                                d[fname].append(n1)
                            d[fname].append(n2)
                        else:
                            d[fname] = [0, n1, n2] if n1 != 0 else [0, n2]
        # print(len(self.phone_list))

    def rate(self, segmentation, filenames, tolerance=2):
        aligned_phones = list(self.phones[fname] for fname in filenames)
        aligned_words  = list(self.words[fname] for fname in filenames)
        phones_score = rate_segmentation(aligned_phones, segmentation, tolerance)
        words_score  = rate_segmentation(aligned_words, segmentation, tolerance)
        return words_score, phones_score


def train_sentencepiece(data, prefix, vocab_size, train=True):
    model_path = str(prefix) + '.model'
    if not ensure_path(model_path):
        if not train:
            raise Exception(f"Tried to segment data, but there is no SentencePiece model at {prefix}. Maybe set train=True?")
        print("Training sentencepiece model...", flush=True)
        SentencePieceTrainer.train(sentence_iterator=iter(data_to_string_arrays(data)), model_prefix=prefix, vocab_size=vocab_size)



def segment_sentencepiece(data, prefix):
    """Segment data using sentencepiece.
    
    Params:

    data: either enumerable of strings (sentences), or enumerable of integer arrays
    prefix: prefix of a model (path/disk location)
    """

    data = data_to_string_arrays(data)
    model_path = str(prefix) + '.model'
    sp = SentencePieceProcessor()
    sp.Load(model_path)
    encodings = sp.Encode(list(data), out_type=str)
    formatted = list(list(word.replace('▁', '').strip() for word in sentence if word.replace('▁', '').strip() != '') for sentence in encodings)
    segmentation = list(get_segmentation(sentence) for sentence in formatted)
    return formatted, segmentation


def segment_viterbi(data, distribution, prefix: pathlib.Path):
    """Segment data using viterbi on center prob distribution
    
    Params:

    data: either enumerable of strings (sentences), or enumerable of integer arrays
    prefix: prefix of a model (path/disk location)
    """
    data = data_to_string_arrays(data)
    vocab_path = str(prefix) + '.vocab'
    lines = [l.split() for l in open(vocab_path)]
    pieces = {pair[0].replace('▁', ' '): float(pair[1]) for pair in lines}
    formatted = []
    
    for i, sentence in tqdm.tqdm(enumerate(data), "Segmenting with viterbi", total=len(data)):
        seg = segment_viterbi_sentence(sentence, distribution[i], pieces)
        formatted.append(seg)

    segmentation = list(get_segmentation(sentence) for sentence in formatted)
    return formatted, segmentation


def segment_viterbi_sentence(sentence, distribution, pieces):

    idx2piece = list(pieces.keys())
    piece2idx = {piece: i for i, piece in enumerate(idx2piece)}
    
    # Build piece_logp matrix
    max_piece_len = max(len(k) for k in pieces.keys())
    T = len(sentence)
    
    piece_logp = np.ones((max_piece_len, T)) * -np.inf
    piece_idx = -np.ones((max_piece_len, T), dtype=np.int64)
    
    # For every length, fill with the most probable piece
    # If there is no piece of this length, skip it - the matrix defaults to -inf values
    for len_ in range(max_piece_len):
        for t in range(T-len_):
            chunk = sentence[t:t+len_+1]
            if chunk in pieces:
                I = np.array(string_to_int_array(chunk))
                piece_logp[len_, t] = pieces[chunk] + distribution[t:t+len_+1, :][np.arange(len_+1), I].sum()
                piece_idx[len_, t] = piece2idx[chunk]
    
    best_lens = sentpiece_viterbi(piece_logp)
    
    # Translate piece lens to actual pieces
    best_pieces = []
    t = 0
    for len_ in best_lens:
        idx = piece_idx[len_-1, t]
        t += len_
        best_pieces.append(idx2piece[idx])
    
    return best_pieces
    

def merge_segmentations(base, internal):
    """Merges two segmentations."""
    return [
        np.array(s1, dtype='int32')[np.array(s2, dtype='int32')-1]
        for s1, s2 in zip(base, internal)
    ]

import os
from collections import defaultdict
from simi.stringify import string_to_int_array
from simi.utils import data_to_string_arrays, ensure_path, get_segmentation
from simi.score import Score

import numpy as np
import tqdm
from pathlib import Path
from numba import jit
from sentencepiece import SentencePieceProcessor, SentencePieceTrainer

from simi.viterbi import sentpiece_viterbi

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
    def __init__(self, name=None, path=None):
        self.phones = defaultdict(list)
        self.words = defaultdict(list)
        self.phone_list = set()
        path = os.path.join(ALIGNMENTS_ROOTPATH, name) if path is None else path
        for csv in Path(path).rglob('*.csv'):
            for line in open(csv, 'r', encoding='utf8'):
                t1, t2, q, kind = line.strip().split(',')
                if kind == 'phones':
                    self.phone_list.add(q)
                n1, n2 = int(round(float(t1)*100)), int(round(float(t2)*100))
                assert n1 < n2, f'line: {line}, n1: {n1}, n2: {n2}'
                d = self.words if kind == 'words' else self.phones
                if csv.stem in d.keys():
                    assert d[csv.stem][-1] <= n1
                    
                    if d[csv.stem][-1] != n1:
                        d[csv.stem].append(n1)
                    d[csv.stem].append(n2)
                else:
                    d[csv.stem] = [0, n1, n2] if n1 != 0 else [0, n2]

    def rate(self, segmentation, filenames, tolerance=2):
        aligned_phones = list(self.phones[fname] for fname in filenames)
        aligned_words  = list(self.words[fname] for fname in filenames)
        phones_score = rate_segmentation(aligned_phones, segmentation, tolerance)
        words_score  = rate_segmentation(aligned_words, segmentation, tolerance)
        return words_score, phones_score


def train_sentencepiece(data, prefix, vocab_size, train=True, max_piece_length=100):
    model_path = str(prefix) + '.model'
    if not ensure_path(model_path):
        if not train:
            raise Exception(f"Tried to segment data, but there is no SentencePiece model at {prefix}. Maybe set train=True?")
        print("Training sentencepiece model...", flush=True)
        SentencePieceTrainer.train(sentence_iterator=iter(data_to_string_arrays(data)), model_prefix=prefix, vocab_size=vocab_size, max_sentencepiece_length=max_piece_length)


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


def segment_viterbi(data, distribution, prefix: Path):
    """Segment data using viterbi on center prob distribution
    
    Params:

    data: either enumerable of strings (sentences), or enumerable of integer arrays
    prefix: prefix of a model (path/disk location)
    """
    data = data_to_string_arrays(data)
    vocab_path = str(prefix) + '.vocab'
    lines = [l.split() for l in open(vocab_path)]
    pieces = {pair[0].replace('▁', ' '): float(pair[1]) for pair in lines}

    # Filter out ' ', <s>, etc.
    pieces = {piece: prob for piece, prob in pieces.items()
              if ' ' not in piece and '<' not in piece}

    formatted = []
   
    # # Merge segments 
    # for i, sentence in tqdm.tqdm(enumerate(data), "Segmenting with viterbi", total=len(data)):
    #     seg = simple_segment_viterbi(sentence, distribution[i], pieces)
    #     formatted.append(seg)

    # Try to improve cluster IDs with SentencePiece
    for i, sentence in tqdm.tqdm(enumerate(data), "Segmenting with viterbi", total=len(data)):
        seg = advanced_segment_viterbi(distribution[i], pieces)
        formatted.append(seg)

    segmentation = list(get_segmentation(sentence) for sentence in formatted)
    return formatted, segmentation


def simple_segment_viterbi(sentence, distribution, pieces):

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


@jit(nopython=True)
def positional_piece_logp_fast(piece2inds, piece_lens, piece_logp, max_piece_len, T, distribution):

    positional_piece_logp = np.ones((max_piece_len+1, T)) * -np.inf
    positional_piece_idx = -np.ones((max_piece_len+1, T), dtype=np.int64)

    for idx in range(piece2inds.shape[0]):

        len_ = piece_lens[idx]

        for t in range(T-len_+1):

            logp = piece_logp[idx]
            for i in range(len_):
                logp += distribution[t+i, piece2inds[idx, i]]

            if logp > positional_piece_logp[len_, t]:
                positional_piece_logp[len_, t] = logp
                positional_piece_idx[len_, t] = idx

    return positional_piece_logp, positional_piece_idx


def positional_piece_logp_slow(piece2idx, max_piece_len, T, pieces, distribution):

    positional_piece_logp = np.ones((max_piece_len+1, T)) * -np.inf
    positional_piece_idx = -np.ones((max_piece_len+1, T), dtype=np.int64)

    pieces_by_len = {l: {} for l in range(1, max_piece_len + 1)}
    for piece, idx in piece2idx.items():
        pieces_by_len[len(piece)][piece] = np.asarray(string_to_int_array(piece))

    # For every length, fill with the most probable piece
    # If there is no piece of this length, skip it - the matrix defaults to -inf values
    for len_ in range(1, max_piece_len + 1):

        if pieces_by_len[len_] == []:
            continue

        for t in range(T-len_+1):
            best_piece = None
            best_logp = -np.inf
            for piece in pieces_by_len[len_]:
                
                I = np.array(string_to_int_array(piece))

                # NOTE try .mean() instead of .sum()
                logp = pieces[piece] + distribution[t:t+len_, :][np.arange(len_), I].sum()

                logp = logp.sum()
                if logp > best_logp:
                    best_logp = logp
                    best_piece = piece

            positional_piece_logp[len_, t] = best_logp
            positional_piece_idx[len_, t] = piece2idx[best_piece]

    return positional_piece_logp, positional_piece_idx


def advanced_segment_viterbi(distribution, pieces):

    # Important: sort by length    
    idx2piece = sorted(list(pieces.keys()), key=len)
    piece2idx = {piece: i for i, piece in enumerate(idx2piece)}

    piece_logp = np.asarray([pieces[p] for p in idx2piece])

    # Build piece_logp matrix
    max_piece_len = max(len(k) for k in pieces.keys())
    T = distribution.shape[0]
    V = len(pieces)

    piece2inds = -np.ones((V, max_piece_len), dtype=np.int64)
    piece_lens = np.zeros((V,), dtype=np.int64)

    for i, piece in enumerate(idx2piece):
        len_ = len(piece)
        piece2inds[i, :len_] = np.asarray(string_to_int_array(piece))
        piece_lens[i] = len_

    positional_piece_logp, positional_piece_idx = positional_piece_logp_fast(
        piece2inds, piece_lens, piece_logp, max_piece_len, T, distribution)

    # positional_piece_logp2, positional_piece_idx2 = positional_piece_logp_slow(
    #     piece2idx, max_piece_len, T, pieces, distribution)
    #
    # assert np.allclose(positional_piece_logp, positional_piece_logp2)
    # assert np.allclose(positional_piece_idx, positional_piece_idx2)

    best_lens = sentpiece_viterbi(positional_piece_logp)

    # Translate piece lens to actual pieces
    best_pieces = []
    t = 0
    for len_ in best_lens:
        idx = positional_piece_idx[len_, t]
        t += len_
        best_pieces.append(idx2piece[idx])

    return best_pieces 


def merge_segmentations(base, internal):
    """Merges two segmentations."""
    return [
        np.array(s1, dtype='int32')[np.array(s2, dtype='int32')-1]
        for s1, s2 in zip(base, internal)
    ]

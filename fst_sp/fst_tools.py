import numpy as np
from kaldi import matrix,decoder
import kaldi.fstext as fst
from utils import EStepRet, PieceCounts, Sentence, SentencePiece, ViterbiPath

#text_to_matrix
#get_lattice
#_to_log_lattice
#viterbi

def text_to_matrix(text, char_symb=None, prepend_space=True):
    if prepend_space:
        text = ' ' + text
    out_np = np.empty((len(text), char_symb.num_symbols()-1))
    out_np.fill(-1e10)

    for i, c in enumerate(text):
        # handle weird SP spacer
        if c == ' ':
            c = '▁'
        c = char_symb.find_index(c)
        out_np[i, c - 1] = 0
    return matrix.Matrix(out_np)

def get_lattice(sentence_text, sp_to_char, prepend_space=True):
    opts = decoder.LatticeFasterDecoderOptions()
    dec = decoder.LatticeFasterDecoder(sp_to_char, opts)
    sentence_mat = text_to_matrix(
        sentence_text, char_symb=sp_to_char.input_symbols(), prepend_space=prepend_space)
    dec.decode(decoder.DecodableMatrixScaled(sentence_mat, 1.0))
    lattice = dec.get_raw_lattice()
    lattice.set_input_symbols(sp_to_char.input_symbols())
    lattice.set_output_symbols(sp_to_char.output_symbols())
    return lattice

def _to_log_lattice(lattice, cfun=lambda x: x.value):
    lattice_log = fst.LogVectorFst()
    for i in range(lattice.num_states()):
        assert lattice_log.add_state() == i
        lattice_log.set_final(i, fst.LogWeight(cfun(lattice.final(i))))
    lattice_log.set_start(lattice.start())
    for i in range(lattice.num_states()):
        for a in lattice.arcs(i):
            lattice_log.add_arc(i, fst.LogArc(
                a.ilabel, a.olabel, fst.LogWeight(cfun(a.weight)), a.nextstate, ))
    return lattice_log

def viterbi(sentence_text, sp_to_char, nshortest=1, normalize_probs=True, prepend_space=True, unigram_weight=1.0):
    def wcfun(w):
        return unigram_weight * w.value1 + w.value2
    lattice_kaldi = get_lattice(
        sentence_text, sp_to_char=sp_to_char, prepend_space=prepend_space)
    if normalize_probs:
        lattice_log = _to_log_lattice(lattice_kaldi, wcfun)
        Z = fst.shortestdistance(lattice_log, reverse=True)[
            lattice_log.start()].value
    else:
        Z = 0.0  # speed hack
    # we will need to do another arc-map here :(
    assert unigram_weight == 1.0
    nbest = fst.shortestpath(lattice_kaldi, nshortest=nshortest)
    # shortestpath fails anyway for Log FST
    weight_zero = fst.LatticeWeight.zero()

    def dump_best_path(state, prefix, log_prob):
        ret = []
        if nbest.final(state) != weight_zero:
            final_log_prob = Z - log_prob - wcfun(nbest.final(state))
            ret = [ViterbiPath(list(prefix), np.exp(
                final_log_prob), final_log_prob)]

        if nbest.num_arcs(state) == 1:
            arc, = nbest.arcs(state)
            if arc.olabel != 0:
                prefix.append(arc.olabel)
            return ret + dump_best_path(arc.nextstate, prefix, log_prob + wcfun(arc.weight))

        for arc in nbest.arcs(state):
            arc_prefix = list(prefix)
            if arc.olabel != 0:
                prefix.append(arc.olabel)
            ret.extend(dump_best_path(arc.nextstate, arc_prefix,
                        log_prob + wcfun(arc.weight)))
        return ret

    return dump_best_path(nbest.start(), [], 0.0)
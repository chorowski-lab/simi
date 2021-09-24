import kaldi.fstext as fst
import numpy as np
from kaldi import decoder, matrix
from kaldi.fstext import SymbolTable

from utils import EStepRet, PieceCounts, Sentence, SentencePiece, ViterbiPath

# text_to_matrix
# get_lattice
# _to_log_lattice
# viterbi


arc_type_to_fst = {
    'log': fst.LogVectorFst,
    'standard': fst.StdVectorFst
}
arc_type_to_arc = {
    'log': fst.LogArc,
    'standard': fst.StdArc
}
arc_type_to_weigth = {
    'log': fst.LogWeight,
    'standard': fst.TropicalWeight
}


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


def get_lattice(sentence_text, sp_to_char, prepend_space=True, lattice_opts=None):
    if lattice_opts is None:
        lattice_opts = decoder.LatticeFasterDecoderOptions()
    dec = decoder.LatticeFasterDecoder(sp_to_char, lattice_opts)
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


def viterbi(sentence_text, sp_to_char, nshortest=1, normalize_probs=True, prepend_space=True, unigram_weight=1.0, lattice_opts=None):
    def wcfun(w):
        return unigram_weight * w.value1 + w.value2
    lattice_kaldi = get_lattice(
        sentence_text, sp_to_char=sp_to_char, prepend_space=prepend_space, lattice_opts=lattice_opts)
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


def compute_piece_counts(sentence_text, sp_to_char, prepend_space=True, unigram_weight=1.0, lattice_opts=None):
    lattice_kaldi = get_lattice(
        sentence_text, sp_to_char=sp_to_char, prepend_space=prepend_space, lattice_opts=lattice_opts)
    # Convert the lattice to log semiring
    # The LatticeWeigth behaves like <Tropical, Tropical> and while it will be faster, the counts will be more approximate
    lattice = _to_log_lattice(
        lattice_kaldi, lambda w: unigram_weight * w.value1 + w.value2)
    alphas = [w.value for w in fst.shortestdistance(lattice)]
    betas = [w.value for w in fst.shortestdistance(lattice, reverse=True)]
    # The cost of reaching the entry node from all terminals - tot cost of the lattice
    Z = betas[lattice.start()]
    counts = {}
    for state in range(lattice.num_states()):
        for arc in lattice.arcs(state):
            if arc.olabel == 0:
                # skip epsilon arcs
                continue
            # prob(reach_state)*prob(exit_from_nextstate)*prob(sumbol_unigram) / sum_allpaths(prob(path))
            log_count = Z - \
                (alphas[state] + betas[arc.nextstate] + arc.weight.value)
            if arc.olabel not in counts:
                counts[arc.olabel] = log_count
            else:
                counts[arc.olabel] = np.logaddexp(
                    counts[arc.olabel], log_count)
    for k in counts:
        counts[k] = np.exp(counts[k])
    return PieceCounts(Z, counts)
# Naive tools - all should be removed when not naive will be equivalent


def text_to_fst(text, arc_type="log", char_symb=None, prepend_space=True):
    if char_symb is None:
        raise NotImplementedError()
    out = arc_type_to_fst[arc_type]()
    out.add_state()
    out.set_start(0)
    out.set_input_symbols(char_symb)
    out.set_output_symbols(char_symb)
    log_one = arc_type_to_weigth[arc_type].one()
    Arc = arc_type_to_arc[arc_type]
    state = 0
    if prepend_space:
        text = ' ' + text
    for c in text:
        # handle weird SP spacer
        if c == ' ':
            c = '▁'
        c = char_symb.find_index(c)
        new_state = out.add_state()
        out.add_arc(state, Arc(c, c, log_one, new_state))
        state = new_state
    out.set_final(state, log_one)
    return out


def get_lattice_naive(sentence_text, sp_to_char, prepend_space=True):
    arc_type = next(iter(sp_to_char.arcs(sp_to_char.start()))).type()
    sentence_fst = text_to_fst(sentence_text, arc_type=arc_type,
                               char_symb=sp_to_char.input_symbols(), prepend_space=prepend_space)
    lattice = fst.compose(sentence_fst, sp_to_char)
    return lattice


def viterbi_naive(sentence_text, sp_to_char, nshortest=1, normalize_probs=True, prepend_space=True):
    lattice = get_lattice_naive(
        sentence_text, sp_to_char=sp_to_char, prepend_space=prepend_space)
    if normalize_probs:
        lattice_log = _to_log_lattice(lattice)
        Z = fst.shortestdistance(lattice_log, reverse=True)[
            lattice_log.start()].value
    else:
        Z = 0.0  # speed hack
    nbest = fst.shortestpath(lattice, nshortest=nshortest)
    # shortestpath fails anyway for Log FST
    weight_zero = fst.TropicalWeight.zero()

    def dump_best_path(state, prefix, log_prob):
        ret = []
        if nbest.final(state) != weight_zero:
            final_log_prob = Z - log_prob - nbest.final(state).value
            ret = [ViterbiPath(list(prefix), np.exp(
                final_log_prob), final_log_prob)]

        if nbest.num_arcs(state) == 1:
            arc, = nbest.arcs(state)
            if arc.olabel != 0:
                prefix.append(arc.olabel)
            return ret + dump_best_path(arc.nextstate, prefix, log_prob + arc.weight.value)

        for arc in nbest.arcs(state):
            arc_prefix = list(prefix)
            if arc.olabel != 0:
                prefix.append(arc.olabel)
            ret.extend(dump_best_path(arc.nextstate,
                       arc_prefix, log_prob + arc.weight.value))
        return ret

    return dump_best_path(nbest.start(), [], 0.0)

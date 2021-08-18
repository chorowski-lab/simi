from collections import defaultdict, namedtuple

import numpy as np
from scipy.special import logsumexp, digamma

# import kaldi.fstext as fst
import openfst_python as fst

Sentence = namedtuple('Sentence', ['text', 'count'])
SentencePiece = namedtuple('SentencePiece', ['index', 'symbol', 'log_freq'])
PieceCounts = namedtuple('PieceCounts', ['Z', 'counts'])
ViterbiPath = namedtuple('ViterbiPath', ['path', 'prob'])
EStepRet = namedtuple('EStepRet', ['objective', 'n_tokens', 'counts'])

def extract_pieces(sp):
    for i in range(sp.vocab_size()):
        yield SentencePiece(i,  sp.id_to_piece(i), sp.GetScore(i))
        

class SentencePieceTrainer:
    def __init__(self, INITIAL_PIECES):
        self._init_symbols(INITIAL_PIECES)
        
    def _init_symbols(self, INITIAL_PIECES):
        # Create symbol tables for FSTs, these can stay constant through training
        CHAR_SYMB = fst.SymbolTable()
        CHAR_SYMB.add_symbol('<eps>')
        for piece in INITIAL_PIECES:
            if piece.log_freq == 0:
                continue
            for char in piece.symbol:
                if CHAR_SYMB.find(char) == -1:
                    CHAR_SYMB.add_symbol(char)

        PIECE_SYMB = fst.SymbolTable()
        PIECE_SYMB.add_symbol('<eps>')
        for piece in INITIAL_PIECES[1:]:
            PIECE_SYMB.add_symbol(piece.symbol, piece.index)
        
        self.CHAR_SYMB = CHAR_SYMB
        self.PIECE_SYMB = PIECE_SYMB
    
    # Create an FST that matches sentencepieces to texts. 
# This one should be pruned after each iteration to speed things up.

    def get_sp_to_char(self, pieces, arc_type='log', piece_symb=None, char_symb=None):
        if piece_symb is None:
            piece_symb = self.PIECE_SYMB
        if char_symb is None:
            char_symb = self.CHAR_SYMB
        sp_to_char = fst.VectorFst(arc_type)
        sp_to_char.set_input_symbols(piece_symb)
        sp_to_char.set_output_symbols(char_symb)
        sp_to_char.add_state()
        sp_to_char.set_start(0)
        log_one = fst.Weight.one(sp_to_char.weight_type())
        sp_to_char.set_final(0, log_one)

        for piece in pieces:
            if piece.log_freq == 0:
                continue
            next_state = 0

            for char in piece.symbol[-1:0:-1]:
                new_state = sp_to_char.add_state()
                sp_to_char.add_arc(new_state, fst.Arc(0, char_symb.find(char), log_one, next_state))
                next_state = new_state
            sp_to_char.add_arc(0, fst.Arc(
                piece.index, char_symb.find(piece.symbol[0]), fst.Weight(sp_to_char.weight_type(), -piece.log_freq), next_state))
        sp_to_char.minimize()
        return sp_to_char

    
    # FST ad lattice helper methods

    def text_to_fst(self, text, arc_type="log", char_symb=None, prepend_space=True):
        if char_symb is None:
            char_symb = self.CHAR_SYMB
        out = fst.VectorFst(arc_type)
        out.add_state()
        out.set_start(0)
        out.set_input_symbols(char_symb)
        out.set_output_symbols(char_symb)
        log_one = fst.Weight.one(out.weight_type())
        state = 0
        if prepend_space:
            text = ' ' + text
        for c in text:
            # handle weird SP spacer
            if c == ' ':
                c = '▁'
            c = char_symb.find(c)
            new_state = out.add_state()
            out.add_arc(state, fst.Arc(c, c, log_one, new_state))
            state = new_state
        out.set_final(state, log_one)
        return out

    def get_lattice(self, sentence_text, sp_to_char, prepend_space=True):
        sentence_fst = self.text_to_fst(sentence_text, arc_type=sp_to_char.arc_type(), char_symb=sp_to_char.output_symbols(), prepend_space=prepend_space)
        lattice = fst.compose(sp_to_char, sentence_fst)
        return lattice


    def compute_piece_counts(self, sentence_text, sp_to_char, prepend_space=True):
        lattice = self.get_lattice(sentence_text, sp_to_char=sp_to_char, prepend_space=prepend_space)
        alphas = [float(w) for w in fst.shortestdistance(lattice)]
        betas = [float(w) for w in fst.shortestdistance(lattice, reverse=True)]
        Z = betas[lattice.start()]  # The cost of reaching the entry node from all terminals - tot cost of the lattice
        counts = defaultdict(float)
        for state in range(lattice.num_states()):
            for arc in lattice.arcs(state):
                if arc.ilabel == 0:
                    # skip epsilon arcs
                    continue
                # prob(reach_state)*prob(exit_from_nextstate)*prob(sumbol_unigram) / sum_allpaths(prob(path))
                count = np.exp(Z - (alphas[state] + betas[arc.nextstate] + float(arc.weight)))
                counts[arc.ilabel] += count
        return PieceCounts(Z, counts)

    def viterbi(self, sentence_text, sp_to_char, nshortest=1, normalize_probs=True, prepend_space=True):
        lattice = self.get_lattice(sentence_text, sp_to_char, prepend_space=prepend_space)
        if normalize_probs:
            Z = float(fst.shortestdistance(fst.arcmap(lattice, map_type='to_log'), reverse=True)[lattice.start()])
        else:
            Z = 0.0  # speed hack
        nbest = fst.shortestpath(lattice, nshortest=nshortest)
        weight_zero = fst.Weight.zero(nbest.weight_type())

        def dump_best_path(state, prefix, log_prob):
            ret = []
            if nbest.final(state) != weight_zero:
                ret = [ViterbiPath(list(prefix), np.exp(Z -log_prob -float(nbest.final(state))))]

            if nbest.num_arcs(state) == 1:
                arc, = nbest.arcs(state)
                if arc.ilabel != 0:
                    prefix.append(arc.ilabel)
                return ret + dump_best_path(arc.nextstate, prefix, log_prob + float(arc.weight))

            for arc in nbest.arcs(state):
                arc_prefix = list(prefix)
                if arc.ilabel != 0:
                    prefix.append(arc.ilabel)
                ret.extend(dump_best_path(arc.nextstate, arc_prefix, log_prob + float(arc.weight)))
            return ret

        return dump_best_path(nbest.start(), [], 0.0)
    
    
    # EM, based on unigram_model_trainer.cc

    def EStep(self, pieces, sentences):
        objective = 0.0
        n_sentences = 0  # all_sentence_freq in sentencepiece
        n_tokens = 0
        sp_to_char = self.get_sp_to_char(pieces)
        sp_to_char_std = fst.arcmap(sp_to_char, map_type='to_std')
        total_counts = defaultdict(float)
        # TODO: parallelize
        for sentence, sent_count in sentences:
            counts = self.compute_piece_counts(sentence, sp_to_char)
            assert np.isfinite(counts.Z)
            objective += counts.Z * sent_count
            n_sentences += sent_count
            # TODO: bug, should incorporate sent_count. This bug seems to be in C++ code too
            n_tokens += len(self.viterbi(sentence, sp_to_char_std)[0][0])
            for symbol, count in counts.counts.items():
                total_counts[symbol] += count * sent_count
        objective /= n_sentences
        return EStepRet(objective, n_tokens, total_counts)

    def MStep(self, pieces, counts, kExpectedFrequencyThreshold = 0.5):
        new_pieces = []
        sum_counts = 0
        for piece in pieces:
            count = counts[piece.index]
            if count < kExpectedFrequencyThreshold:
                continue
            new_pieces.append(SentencePiece(piece.index, piece.symbol, count))
            sum_counts += count

        log_sum = digamma(sum_counts)
        new_pieces = [SentencePiece(piece.index, piece.symbol, digamma(piece.log_freq) - log_sum) for piece in new_pieces]
        return new_pieces

    def prune_pieces(self, pieces, sentences, desired_size, prune_frac):
        sp_to_char = self.get_sp_to_char(pieces, arc_type='standard')

        always_keep = {}
        alternatives = {}
        for piece in pieces:
            nbest = self.viterbi(piece.symbol, sp_to_char, nshortest=2, normalize_probs=False, prepend_space=False)
            if len(nbest) == 1:
                always_keep[piece.index] = True
                continue
            if len(nbest[0].path) > 1:
                # The best tokenization for this piece is not itself!
                always_keep[piece.index] = False
                continue
            always_keep[piece.index] = True
            alternatives[piece.index] = nbest[1].path

        inverted = defaultdict(list)
        piece_usage_counts = defaultdict(float)
        v_sums = 0
        for sent_i, (sentence, sent_count) in enumerate(sentences):
            v_sums += sent_count
            nbest, = self.viterbi(sentence, sp_to_char)
            for piece in nbest.path:
                inverted[piece].append(sent_i)
                piece_usage_counts[piece] += sent_count

        usage_sum = sum(piece_usage_counts.values())
        log_usage_sum = np.log(usage_sum)

        new_pieces = []
        candidates = []
        for piece in pieces:
            p_id = piece.index
            if piece_usage_counts[p_id] == 0 and not always_keep[p_id]:
                continue
            if not alternatives.get(p_id):
                new_pieces.append(piece)
                continue

            F = sum(sentences[sent].sent_count for sent in inverted[p_id]) / v_sums  # TODO: add sentence weigths
            logprob_sp = np.log(piece_usage_counts[p_id]) - log_usage_sum

            # sum + freq[i] * (alternatives.size() - 1)
            logsum_alt = np.log(usage_sum + piece_usage_counts[p_id] * (len(pieces) - 1))
            # seems to be a bug - alternatives.size() == num_sentencepieces (!)
            # I presume they wanted this:
            # logsum_alt = np.log(usage_sum + piece_usage_counts[p_id] * (len(alternatives[p_id]) - 1))

            logprob_alt = 0
            for alt_piece_id in alternatives[p_id]:
                logprob_alt += np.log(piece_usage_counts[alt_piece_id] + piece_usage_counts[p_id]) - logsum_alt
            loss = F * (logprob_sp - logprob_alt)

            candidates.append((loss, piece))

        pruned_size = max(desired_size, int(len(pieces) * prune_frac))
        new_pieces.extend([piece for _, piece in sorted(candidates)[:pruned_size - len(new_pieces)]])

        return new_pieces

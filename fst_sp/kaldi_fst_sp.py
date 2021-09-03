from collections import defaultdict, namedtuple

import numpy as np
from scipy.special import logsumexp, digamma

import kaldi.fstext as fst
from kaldi import matrix, decoder
# import openfst_python as fst

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
        self._unk_penalty = 10
        self._reproduce_unk_bug = True
        # self._reproduce_counting_bug = True
        
    def _init_symbols(self, INITIAL_PIECES):
        # Create symbol tables for FSTs, these can stay constant through training
        CHAR_SYMB = fst.SymbolTable()
        CHAR_SYMB.add_symbol('<eps>')
        for piece in INITIAL_PIECES:
            if piece.log_freq == 0:
                continue
            for char in piece.symbol:
                if CHAR_SYMB.find_index(char) == -1:
                    CHAR_SYMB.add_symbol(char)

        PIECE_SYMB = fst.SymbolTable()
        PIECE_SYMB.add_symbol('<eps>')
        for piece in INITIAL_PIECES[1:]:
            PIECE_SYMB.add_pair(piece.symbol, piece.index)
        PIECE_SYMB.add_symbol('<unk>')
        
        self.CHAR_SYMB = CHAR_SYMB
        self.PIECE_SYMB = PIECE_SYMB
    
    # Create an FST that matches sentencepieces to texts. 
    # This one should be pruned after each iteration to speed things up.
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
    
    def get_sp_to_char(self, pieces, arc_type='log', piece_symb=None, char_symb=None):
        if piece_symb is None:
            piece_symb = self.PIECE_SYMB
        if char_symb is None:
            char_symb = self.CHAR_SYMB
        
        required_pieces = {
            char_symb.find_symbol(i) for i in range(1, char_symb.num_symbols())
        }
        
        sp_to_char = self.arc_type_to_fst[arc_type]()
        Weight = self.arc_type_to_weigth[arc_type]
        Arc = self.arc_type_to_arc[arc_type]
        sp_to_char.set_output_symbols(piece_symb)
        sp_to_char.set_input_symbols(char_symb)
        sp_to_char.add_state()
        sp_to_char.set_start(0)
        log_one = Weight.one()
        sp_to_char.set_final(0, log_one)
        
        prefix2state = [('', 0)]
        for piece in sorted(pieces, key=lambda x: x.symbol):
            if piece.log_freq == 0:
                continue
            assert piece.symbol == piece_symb.find_symbol(piece.index)
            
            required_pieces.discard(piece.symbol)
            
            while not piece.symbol.startswith(prefix2state[-1][0]):
                prefix2state.pop()
            state = prefix2state[-1][1]
            for i in range(len(prefix2state[-1][0]), len(piece.symbol)-1):
            # for char in piece.symbol[len(prefix2state[-1][0]):-1]:
                char = piece.symbol[i]
                new_state = sp_to_char.add_state()
                prefix2state.append((piece.symbol[:i+1], new_state))
                sp_to_char.add_arc(state, Arc(char_symb.find_index(char), 0, log_one, new_state))
                state = new_state
            sp_to_char.add_arc(state, Arc(
                char_symb.find_index(piece.symbol[-1]), piece.index, Weight(-piece.log_freq), 0))
        
        unk_penalty = min(pieces, key=lambda x:x.log_freq).log_freq - self._unk_penalty
        if self._reproduce_unk_bug:
            unk_id = min(pieces, key=lambda x:x.index).index
        else:
            unk_id = piece_symb.find_index('<unk>')
        
        for symbol in required_pieces:
            sp_to_char.add_arc(0, Arc(
                char_symb.find_index(symbol), unk_id, Weight(-unk_penalty), 0))
        return sp_to_char
    
    # Kaldi decoder based helper methods
    
    def text_to_matrix(self, text, char_symb=None, prepend_space=True):
        if char_symb is None:
            char_symb = self.CHAR_SYMB
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
    
    def get_lattice(self, sentence_text, sp_to_char, prepend_space=True):
        opts = decoder.LatticeFasterDecoderOptions()
        dec = decoder.LatticeFasterDecoder(sp_to_char, opts)
        sentence_mat = self.text_to_matrix(sentence_text, char_symb=sp_to_char.input_symbols(), prepend_space=prepend_space)
        dec.decode(decoder.DecodableMatrixScaled(sentence_mat, 1.0))
        lattice = dec.get_raw_lattice()
        lattice.set_input_symbols(sp_to_char.input_symbols())
        lattice.set_output_symbols(sp_to_char.output_symbols())
        return lattice
    
    def _to_log_lattice(self, lattice, cfun=lambda x: x.value):
        lattice_log = fst.LogVectorFst()
        for i in range(lattice.num_states()):
            assert lattice_log.add_state() == i
            lattice_log.set_final(i, fst.LogWeight(cfun(lattice.final(i))))
        lattice_log.set_start(lattice.start())
        for i in range(lattice.num_states()):
            for a in lattice.arcs(i):
                lattice_log.add_arc(i, fst.LogArc(a.ilabel, a.olabel, fst.LogWeight(cfun(a.weight)), a.nextstate, ))
        return lattice_log

    def compute_piece_counts(self, sentence_text, sp_to_char, prepend_space=True, unigram_weight=1.0):
        lattice_kaldi = self.get_lattice(sentence_text, sp_to_char=sp_to_char, prepend_space=prepend_space)
        # Convert the lattice to log semiring
        # The LatticeWeigth behaves like <Tropical, Tropical> and while it will be faster, the counts will be more approximate
        lattice = self._to_log_lattice(lattice_kaldi, lambda w: unigram_weight * w.value1 + w.value2)
        alphas = [w.value for w in fst.shortestdistance(lattice)]
        betas = [w.value for w in fst.shortestdistance(lattice, reverse=True)]
        Z = betas[lattice.start()]  # The cost of reaching the entry node from all terminals - tot cost of the lattice
        counts = {}
        for state in range(lattice.num_states()):
            for arc in lattice.arcs(state):
                if arc.olabel == 0:
                    # skip epsilon arcs
                    continue
                # prob(reach_state)*prob(exit_from_nextstate)*prob(sumbol_unigram) / sum_allpaths(prob(path))
                log_count = Z - (alphas[state] + betas[arc.nextstate] + arc.weight.value)
                if arc.olabel not in counts:
                    counts[arc.olabel] = log_count
                else:
                    counts[arc.olabel] = np.logaddexp(counts[arc.olabel], log_count)
        for k in counts:
            counts[k] = np.exp(counts[k])
        return PieceCounts(Z, counts)

    def viterbi(self, sentence_text, sp_to_char, nshortest=1, normalize_probs=True, prepend_space=True, unigram_weight=1.0):
        def wcfun(w):
            return unigram_weight * w.value1 + w.value2
        lattice_kaldi = self.get_lattice(sentence_text, sp_to_char=sp_to_char, prepend_space=prepend_space)
        if normalize_probs:
            lattice_log = self._to_log_lattice(lattice_kaldi, wcfun)
            Z = fst.shortestdistance(lattice_log, reverse=True)[lattice_log.start()].value
        else:
            Z = 0.0  # speed hack
        assert unigram_weight == 1.0  # we will need to do another arc-map here :(
        nbest = fst.shortestpath(lattice_kaldi, nshortest=nshortest)
        # shortestpath fails anyway for Log FST
        weight_zero = fst.LatticeWeight.zero()

        def dump_best_path(state, prefix, log_prob):
            ret = []
            if nbest.final(state) != weight_zero:
                ret = [ViterbiPath(list(prefix), np.exp(Z -log_prob -wcfun(nbest.final(state))))]

            if nbest.num_arcs(state) == 1:
                arc, = nbest.arcs(state)
                if arc.olabel != 0:
                    prefix.append(arc.olabel)
                return ret + dump_best_path(arc.nextstate, prefix, log_prob + wcfun(arc.weight))

            for arc in nbest.arcs(state):
                arc_prefix = list(prefix)
                if arc.olabel != 0:
                    prefix.append(arc.olabel)
                ret.extend(dump_best_path(arc.nextstate, arc_prefix, log_prob + wcfun(arc.weight)))
            return ret

        return dump_best_path(nbest.start(), [], 0.0)

    
    # naive FST and lattice helper methods

    def text_to_fst(self, text, arc_type="log", char_symb=None, prepend_space=True):
        if char_symb is None:
            char_symb = self.CHAR_SYMB
        out = self.arc_type_to_fst[arc_type]()
        out.add_state()
        out.set_start(0)
        out.set_input_symbols(char_symb)
        out.set_output_symbols(char_symb)
        log_one = self.arc_type_to_weigth[arc_type].one()
        Arc = self.arc_type_to_arc[arc_type]
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

    def get_lattice_naive(self, sentence_text, sp_to_char, prepend_space=True):
        arc_type = next(iter(sp_to_char.arcs(sp_to_char.start()))).type()
        sentence_fst = self.text_to_fst(sentence_text, arc_type=arc_type, 
                                        char_symb=sp_to_char.input_symbols(), prepend_space=prepend_space)
        lattice = fst.compose(sentence_fst, sp_to_char)
        return lattice

    def compute_piece_counts_naive(self, sentence_text, sp_to_char, prepend_space=True):
        lattice = self.get_lattice_naive(sentence_text, sp_to_char=sp_to_char, prepend_space=prepend_space)
        alphas = [w.value for w in fst.shortestdistance(lattice)]
        betas = [w.value for w in fst.shortestdistance(lattice, reverse=True)]
        Z = betas[lattice.start()]  # The cost of reaching the entry node from all terminals - tot cost of the lattice
        counts = {}
        for state in range(lattice.num_states()):
            for arc in lattice.arcs(state):
                if arc.olabel == 0:
                    # skip epsilon arcs
                    continue
                # prob(reach_state)*prob(exit_from_nextstate)*prob(sumbol_unigram) / sum_allpaths(prob(path))
                log_count = Z - (alphas[state] + betas[arc.nextstate] + arc.weight.value)
                if arc.olabel not in counts:
                    counts[arc.olabel] = log_count
                else:
                    counts[arc.olabel] = np.logaddexp(counts[arc.olabel], log_count)
        for k in counts:
            counts[k] = np.exp(counts[k])
        return PieceCounts(Z, counts)

    def viterbi_naive(self, sentence_text, sp_to_char, nshortest=1, normalize_probs=True, prepend_space=True):
        lattice = self.get_lattice_naive(sentence_text, sp_to_char=sp_to_char, prepend_space=prepend_space)
        if normalize_probs:
            lattice_log = self._to_log_lattice(lattice)
            Z = fst.shortestdistance(lattice_log, reverse=True)[lattice_log.start()].value
        else:
            Z = 0.0  # speed hack
        nbest = fst.shortestpath(lattice, nshortest=nshortest)
        # shortestpath fails anyway for Log FST
        weight_zero = fst.TropicalWeight.zero()

        def dump_best_path(state, prefix, log_prob):
            ret = []
            if nbest.final(state) != weight_zero:
                ret = [ViterbiPath(list(prefix), np.exp(Z -log_prob -nbest.final(state).value))]

            if nbest.num_arcs(state) == 1:
                arc, = nbest.arcs(state)
                if arc.olabel != 0:
                    prefix.append(arc.olabel)
                return ret + dump_best_path(arc.nextstate, prefix, log_prob + arc.weight.value)

            for arc in nbest.arcs(state):
                arc_prefix = list(prefix)
                if arc.olabel != 0:
                    prefix.append(arc.olabel)
                ret.extend(dump_best_path(arc.nextstate, arc_prefix, log_prob + arc.weight.value))
            return ret

        return dump_best_path(nbest.start(), [], 0.0)
    
    
    # EM, based on unigram_model_trainer.cc

    def EStep(self, pieces, sentences):
        objective = 0.0
        n_sentences = 0  # all_sentence_freq in sentencepiece
        n_tokens = 0
        sp_to_char_std = self.get_sp_to_char(pieces, 'standard')
        total_counts = defaultdict(float)
        # TODO: parallelize
        for sentence, sent_count in sentences:
            counts = self.compute_piece_counts(sentence, sp_to_char_std)
            assert np.isfinite(counts.Z)
            objective += counts.Z * sent_count
            n_sentences += sent_count
            # TODO: bug, should incorporate sent_count. This bug seems to be in C++ code too
            # please check self._reproduce_counting_bug
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

#             if len(piece.symbol) == 1:
#                 new_pieces.append(SentencePiece(piece.index, piece.symbol, count))
#                 sum_counts += count
#                 continue

            if count < kExpectedFrequencyThreshold:
                continue
            new_pieces.append(SentencePiece(piece.index, piece.symbol, count))
            sum_counts += count

        log_sum = digamma(sum_counts)
        new_pieces = [SentencePiece(piece.index, piece.symbol, digamma(piece.log_freq) - log_sum if np.isfinite(digamma(piece.log_freq) - log_sum) else -1e3) for piece in new_pieces]
        return new_pieces

    def prune_pieces(self, pieces, sentences, desired_size, prune_frac):
        sp_to_char = self.get_sp_to_char(pieces, arc_type='standard')

        always_keep = {}
        alternatives = {}
        for piece in pieces:
            nbest = self.viterbi_naive(piece.symbol, sp_to_char, nshortest=2, normalize_probs=False, prepend_space=False)
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
            if not alternatives.get(p_id):
                new_pieces.append(piece)
                continue

            if piece_usage_counts[p_id] == 0 and not always_keep[p_id]:
                continue

            F = sum(sentences[sent].count for sent in inverted[p_id]) / v_sums  # TODO: add sentence weigths
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

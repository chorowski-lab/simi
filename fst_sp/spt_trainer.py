from collections import Counter, defaultdict

import kaldi.fstext as fst
import numpy as np
from scipy.special import digamma, logsumexp

from esa import ESA
from spt_model import SentencePieceModel
import fst_tools
from utils import EStepRet, PieceCounts, Sentence, SentencePiece, ViterbiPath


class SentencePieceTrainerParameters(object):

    defaults = {
        "vocab_size": 90,
        "max_piece_length": 1000,
        'num_sub_iterations': 2,  # EM subiterations
        'seed_vocab_size': 10000000,
        'verbose': False,  # more information printed
    }

    def __init__(self, parameters=None):
        if parameters:
            raise NotImplementedError()

        self.vocab_size = SentencePieceTrainerParameters.defaults["vocab_size"]
        self.max_piece_length = SentencePieceTrainerParameters.defaults["max_piece_length"]
        self.num_sub_iterations = SentencePieceTrainerParameters.defaults["num_sub_iterations"]
        self.verbose = SentencePieceTrainerParameters.defaults["verbose"]
        self.seed_vocab_size = SentencePieceTrainerParameters.defaults["seed_vocab_size"]


class SentencePieceTrainer(object):
    @staticmethod
    def get_seed_pieces(sentences, seed_vocab_size, max_piece_length, debug=False):
        def to_log_prob(pieces):
            Z = np.log(sum(score for p, score in pieces))
            pieces = [(p, np.log(score) - Z) for p, score in pieces]
            return pieces
        print("Extracting frequent sub strings...")

        # Makes an enhanced suffix array to extract all sub strings occurring
        # more than 2 times in the sentence.
        
        delimiter=u'\u25C6'

        esa = ESA()
        esa.fit([s+delimiter for s,c in sentences], delimiter=delimiter,
                max_piece_len=max_piece_length, debug=debug)

        seed_sentp = sorted(esa.pieces(), key=lambda p_score: -p_score[1])

        # TODO: bug(?) - single letters (all?) are added by esa, temporary workaround:
        seed_sentp = [x for x in seed_sentp if len(x[0]) > 1]

        # Prune
        seed_sentp = seed_sentp[:seed_vocab_size]

        # all_chars must be included in the seed sentencepieces.
        all_chars = Counter()
        for (s,c) in sentences:
            all_chars.update(s)
        del all_chars[delimiter]

        print(f"Added {len(all_chars)} characters to seed vocabulary")
        for c, cnt in all_chars.items():
            seed_sentp.append((c, cnt))  # 0.5)) # XXX XXX XXX

        seed_sentp = to_log_prob(seed_sentp)
        seed_sentp = sorted(seed_sentp, key=lambda p_score: -p_score[1])
        seed_sentp.insert(0, ("<unk>", 0))
        seed_sentp.insert(1, ("<s>", 0)) #unused so far
        seed_sentp.insert(2, ("</s>", 0))

        print(f"Initialized {len(seed_sentp)} seed sentencepieces")
        return [SentencePiece(ind, symb, freq) for ind, (symb, freq) in enumerate(seed_sentp)]

    # I changed the way it works - instead of handling model saving, it simply returns it.
    # Saving will be done by a model itself, I think it's a bit simpler solution.
    #
    # The 'sentences' param should be the input/training data, in form of list(str) or list(list(int)).
    # Maybe we should also consider numpy array?
    #
    # Also, the parameters are subjects for change (specifically addition).
    @staticmethod
    def train(sentences, vocab_size=90, prune_fact=0.85, num_subiter=2, seed_vocab_size=10000000, max_piece_length=100, verbose=False) -> SentencePieceModel:

        types = [type(s) for s in sentences]

        if all(t == str for t in types):
            sentences = [Sentence((" " + s.strip()).replace(" ", "▁"), 1)
                         for s in sentences]  # _This_format_of_sequence
        elif all(t == Sentence for t in types):
            sentences = [Sentence((" " + s.strip()).replace(" ", "▁"), c)
                         for (s, c) in sentences]  # _This_format_of_sequence
        else:
            raise NotImplementedError()

        pieces = SentencePieceTrainer.get_seed_pieces(sentences,  # In esa
                                                       seed_vocab_size=seed_vocab_size,
                                                       max_piece_length=max_piece_length,
                                                       debug=verbose)

        # Sentencepiece training
        T = SentencePieceTrainer(pieces)  

        while True:
            for sub_iter in range(num_subiter):
                e_ret = T.EStep(pieces, sentences)
                pieces = T.MStep(pieces, e_ret.counts)
                print(f"EM sub_iter={sub_iter} size={len(pieces)} tot_piece_prob={np.exp(logsumexp([piece.log_freq for piece in pieces]))} "
                      f"obj={e_ret.objective} num_tokens={e_ret.n_tokens} num_tokens/piece={e_ret.n_tokens / len(pieces)}")

            if len(pieces) <= vocab_size:
                break

            pieces = T.prune_pieces(pieces, sentences, vocab_size, prune_fact)

            if len(pieces) <= vocab_size:
                break

        pieces = sorted(pieces, key=lambda x: -x.log_freq)
        pieces.insert(0, SentencePiece(0,"<unk>", 0))
        pieces.insert(1, SentencePiece(0,"<s>", 0)) #unused so far
        pieces.insert(2, SentencePiece(0,"</s>", 0))
        pieces = [ SentencePiece(i, sentence, log_freq) for i,(_,sentence,log_freq) in enumerate(pieces)]

        T2 = SentencePieceTrainer(pieces)
        sp_to_char = T2.get_sp_to_char(pieces,'standard')

        return SentencePieceModel(model=sp_to_char, vocab=pieces)

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
            if not piece.symbol == piece_symb.find_symbol(piece.index):
                print(piece.symbol , piece_symb.find_symbol(piece.index))
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
                sp_to_char.add_arc(state, Arc(
                    char_symb.find_index(char), 0, log_one, new_state))
                state = new_state
            sp_to_char.add_arc(state, Arc(
                char_symb.find_index(piece.symbol[-1]), piece.index, Weight(-piece.log_freq), 0))

        unk_penalty = min(
            pieces, key=lambda x: x.log_freq).log_freq - self._unk_penalty
        if self._reproduce_unk_bug:
            unk_id = min(pieces, key=lambda x: x.index).index
        else:
            unk_id = piece_symb.find_index('<unk>')

        for symbol in required_pieces:
            sp_to_char.add_arc(0, Arc(
                char_symb.find_index(symbol), unk_id, Weight(-unk_penalty), 0))
        return sp_to_char

    # Kaldi decoder based helper methods

    def compute_piece_counts(self, sentence_text, sp_to_char, prepend_space=True, unigram_weight=1.0):
        lattice_kaldi = fst_tools.get_lattice(
            sentence_text, sp_to_char=sp_to_char, prepend_space=prepend_space)
        # Convert the lattice to log semiring
        # The LatticeWeigth behaves like <Tropical, Tropical> and while it will be faster, the counts will be more approximate
        lattice = fst_tools._to_log_lattice(
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
            n_tokens += len(fst_tools.viterbi(sentence, sp_to_char_std)[0][0])
            for symbol, count in counts.counts.items():
                total_counts[symbol] += count * sent_count
        objective /= n_sentences
        return EStepRet(objective, n_tokens, total_counts)

    def MStep(self, pieces, counts, kExpectedFrequencyThreshold=0.5):
        new_pieces = []
        sum_counts = 0
        for piece in pieces:
            count = counts[piece.index]

            # if len(piece.symbol) == 1:
            #     new_pieces.append(SentencePiece(piece.index, piece.symbol, count))
            #     sum_counts += count
            #     continue

            if count < kExpectedFrequencyThreshold:
                continue
            new_pieces.append(SentencePiece(piece.index, piece.symbol, count))
            sum_counts += count

        log_sum = digamma(sum_counts)
        new_pieces = [SentencePiece(piece.index, piece.symbol, digamma(
            piece.log_freq) - log_sum if np.isfinite(digamma(piece.log_freq) - log_sum) else -1e3) for piece in new_pieces]
        return new_pieces

    def prune_pieces(self, pieces, sentences, desired_size, prune_frac):
        sp_to_char = self.get_sp_to_char(pieces, arc_type='standard')

        always_keep = {}
        alternatives = {}
        for piece in pieces:
            nbest = fst_tools.viterbi(
                piece.symbol, sp_to_char, nshortest=2, normalize_probs=False, prepend_space=False)
            if len(nbest) == 1:
                #                 print(f'PP {piece.symbol} always keep true FS {nbest[0].log_prob}')
                always_keep[piece.index] = True
                continue
            if len(nbest[0].path) > 1:
                # The best tokenization for this piece is not itself!
                #                 print(f'PP {piece.symbol} always keep false FS {nbest[0].log_prob}/{nbest[1].log_prob}')
                always_keep[piece.index] = False
                continue
            always_keep[piece.index] = True
            #print(f'PP {piece.symbol} always keep true FS {nbest[0].log_prob}/{nbest[1].log_prob} alternatives {[self.PIECE_SYMB.find_symbol(pi) for pi in nbest[1].path]}')
            alternatives[piece.index] = nbest[1].path

        inverted = defaultdict(list)
        piece_usage_counts = defaultdict(float)
        v_sums = 0
        for sent_i, (sentence, sent_count) in enumerate(sentences):
            v_sums += sent_count
            nbest, = fst_tools.viterbi(sentence, sp_to_char)
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
                #print(f'{p_id}/{piece.symbol} accepted for lack of alternatives')
                continue

            if piece_usage_counts[p_id] == 0 and not always_keep[p_id]:
                #                 print(f'{p_id}/{piece.symbol} discarded')
                continue

            # TODO: add sentence weigths
            F = sum(sentences[sent].count for sent in inverted[p_id]) / v_sums
            logprob_sp = np.log(piece_usage_counts[p_id]) - log_usage_sum

            # sum + freq[i] * (alternatives.size() - 1)
            logsum_alt = np.log(
                usage_sum + piece_usage_counts[p_id] * (len(pieces) - 1))
            # seems to be a bug - alternatives.size() == num_sentencepieces (!)
            # I presume they wanted this:
            # logsum_alt = np.log(usage_sum + piece_usage_counts[p_id] * (len(alternatives[p_id]) - 1))

            logprob_alt = 0
            for alt_piece_id in alternatives[p_id]:
                logprob_alt += np.log(
                    piece_usage_counts[alt_piece_id] + piece_usage_counts[p_id]) - logsum_alt
            loss = F * (logprob_sp - logprob_alt)
            #print(f'{p_id}/{piece.symbol} loss {loss}')

            candidates.append((-loss, piece))

        pruned_size = max(desired_size, int(len(pieces) * prune_frac))
        new_pieces.extend([piece for _, piece in sorted(
            candidates)[:pruned_size - len(new_pieces)]])

        return new_pieces

# for development testing
if __name__=="__main__":
    # list of strings
    string_data = list(line.strip() for line in open('./data/botchan_small.txt', 'r', encoding='utf8'))
    # train model
    model = SentencePieceTrainer.train(string_data)
    
    # example sentence (string)
    sentences = ['I saw a girl with a telescope.']
    # segmenting the sentence
    encoding = model.encode(sentences)
    print(encoding)
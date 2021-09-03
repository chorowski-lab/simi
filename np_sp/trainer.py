from collections import Counter

import numpy as np

import pygtrie

from esa import ESA


print_ = print


def logaddexp(x, y, initial=True):
    if initial:
        return y  # log(exp(y)) == y
    else:
        return np.logaddexp(x, y)


def to_log_prob(pieces):
    Z = np.log(sum(score for p, score in pieces))
    print(sorted([s for p,s in pieces]))
    pieces = [(p, np.log(score) - Z) for p, score in pieces]
    return pieces


def digamma(x):
    ret = 0
    while (x < 7):
        ret -= 1/x
        x += 1
    x -= 0.5
    xx = 1.0 / x
    xx2 = xx ** 2
    xx4 = xx2 ** 2
    ret += (np.log(x) + (1.0 / 24.0) * xx2 - (7.0 / 960.0) * xx4
            + (31.0 / 8064.0) * xx4 * xx2 - (127.0 / 30720.0) * xx4 * xx4)
    return ret


class Lattice:

    def __len__(self):
        # Subtract 1 for BOS
        return sum(len(nodes_at_pos) for nodes_at_pos in self.beg_nodes) - 1

    def set_sentence(self, sentence):

        # surface = sentence  # XXX copy-paste the sequence ?
        self.sentence = sentence
        self.N = len(sentence)

        self.beg_nodes = [[] for _ in range(self.N + 1)]
        self.end_nodes = [[] for _ in range(self.N + 1)]

        # Add BOS and EOS
        self.end_nodes[0].append(-1)  #{'id': -1, 'pos': 0})
        self.beg_nodes[-1].append(-1)  #{'id': -1, 'pos': self.N})

    def populate_nodes(self, model):

        self.node2piece = {}

        node = 0
        for pos in range(self.N):
            prefs = 0
            for piece, piece_len in model.prefixes(self.sentence[pos:]):
                self.beg_nodes[pos].append(node)
                self.end_nodes[pos + piece_len].append(node)
                self.node2piece[node] = piece
                node += 1
                prefs += 1
            if prefs == 0:
                print(f'Did not found pieces to match at {pos}: #{self.sentence[pos:]}')
                import ipdb
                ipdb.set_trace()
                raise ValueError

    def populate_marginal(self, freq, scores, expected):

        n_nodes = len(self)
        alpha = np.zeros((n_nodes,))
        beta = np.zeros((n_nodes,))

        for pos in range(self.N + 1):
            for rnode in self.beg_nodes[pos]:
                for i, lnode in enumerate(self.end_nodes[pos]):

                    if lnode == -1:  # BOS, EOS
                        score = 0
                    else:
                        score = scores[self.node2piece[lnode]]
                    alpha[rnode] = logaddexp(alpha[rnode],
                                             alpha[lnode] + score,
                                             i == 0)
        for pos in range(self.N, -1, -1):
            for lnode in self.end_nodes[pos]:
                for i, rnode in enumerate(self.beg_nodes[pos]):

                    if rnode == -1:  # BOS, EOS
                        score = 0
                    else:
                        score = scores[self.node2piece[rnode]]
                    beta[lnode] = logaddexp(beta[lnode],
                                            beta[rnode] + score,
                                            i == 0)
        Z = alpha[self.beg_nodes[self.N][0]]
        for pos in range(self.N):
            for node in self.beg_nodes[pos]:
                assert node >= 0
                piece = self.node2piece[node]
                expected[piece] += freq * np.exp(
                    alpha[node] + beta[node] + scores[piece] - Z)

        return freq * Z

    def viterbi_iter(self, scores, piece_lens, model):

        backtrace_score = np.zeros((self.N + 1,))  #, value=0)  #-np.inf)
        prev_node = np.zeros((self.N + 1,))

        for pos in range(1, self.N):  # XXX self.N + 1 ?
            # print('pos:', pos, 'len(beg_nodes):', len(self.beg_nodes[pos]),
            #       'len(end_nodes):', len(self.end_nodes[pos]))

            for rnode in self.beg_nodes[pos]:


                best_score = -np.inf
                best_node = None

                for lnode in self.end_nodes[pos]:

                    if lnode == -1:
                        continue

                    lpiece = self.node2piece[lnode]
                    lnode_len = piece_lens[lpiece]
                    back_score = backtrace_score[pos - lnode_len]
                    score = back_score + scores[lpiece]

                    if score > best_score:
                        best_node = lnode
                        best_score = score

                if best_node is None:
                    import ipdb
                    ipdb.set_trace()

                assert best_node is not None, \
                    "Failed to find the best path in Viterbi.";

            prev_node[pos] = best_node
            backtrace_score[pos] = best_score

        best_path = -np.ones((self.N,), dtype=np.int64)
        pos = self.N
        for i in range(self.N):
            best_path[i] = prev_node[pos]
            pos -= piece_lens[ self.node2piece[prev_node[pos]] ]
            if pos <= 0:
                break

        return best_path[:i+1][::-1]


class Model:

    def __init__(self):
        self.trie = None
        self.scores = None
        # self.max_piece_len = -1

    def build_trie(self, sentences, pieces, with_vocab=True, delimiter='$'):
        self.trie = pygtrie.CharTrie()
        self.max_piece_len = 0
        self.pieces = []
        self.scores = []
        self.piece_lens = []

        for piece, score in pieces:
            if piece != '':

                assert np.isfinite(score)

                self.pieces.append(piece)
                self.scores.append(score)
                self.piece_lens.append(len(piece))

                id_ = len(self.pieces) - 1
                self.trie[piece] = id_


        # self.max_piece_len = max(self.piece_lens)

        if with_vocab:

            DEFAULT_SCORE = 0.5  # XXX

            vocab = set()
            for s in sentences:
                vocab.update(s)

            for sym in vocab:
                if not sym in self.trie:
                    self.pieces.append(sym)
                    self.scores.append(DEFAULT_SCORE)
                    self.piece_lens.append(1)

                    id_ = len(self.pieces) - 1
                    self.trie[piece] = id_

    def prefixes(self, s):
        for p, id_ in self.trie.prefixes(s):
            yield id_, len(self.pieces[id_])

    def get_piece_size(self):
        return 1000.0  # XXX


class Trainer:

    def make_seed_sentence_pieces(self, sentences, num_seed_sentpieces,
                                  delimiter='#'):

        # corpus = delimiter.join(sentences)
        corpus = sentences

        print_("Extracting frequent sub strings...")

        # Makes an enhanced suffix array to extract all sub strings occurring
        # more than 2 times in the sentence.
        esa = ESA()
        esa.fit(corpus, delimiter=delimiter, max_piece_len=4)

        # seed_sentp = {pc: (freq, score) for (pc, freq, score) in esa.pieces()}
        # # Sort by the coverage of sub strings.
        # seed_sentp = sorted(seed_sentp.items(),
        #                     key=lambda p, (freq, score): -score)

        # seed_sentp = {pc: score for (pc, freq, score) in esa.pieces()}
        seed_sentp = sorted(esa.pieces(), key=lambda p_score: -p_score[1])
        # seed_sentp = sorted(seed_sentp.items(), key=lambda p, score: -score)

        # Prune
        seed_sentp = seed_sentp[:num_seed_sentpieces]

        # all_chars must be included in the seed sentencepieces.
        all_chars = Counter()
        for s in corpus:
            all_chars.update(s)
        del all_chars[delimiter]

        for c, cnt in all_chars.items():
            # seed_sentp.append((c, count)) # XXX XXX XXX
            seed_sentp.append((c, cnt))  # 0.5)) # XXX XXX XXX

        seed_sentp = to_log_prob(seed_sentp)

        print_(f"Initialized {len(seed_sentp)} seed sentencepieces")

        return seed_sentp

    def run_e_step(self, model, sentences):

        # all_sentence_freq = sum(freq for _, freq in sentences)
        all_sentence_freq = len(sentences)  # Assumes sentences are unique

        lattice = Lattice()
        # expected = np.zeros((model.get_piece_size(),))
        obj = 0
        ntokens = 0

        expected = np.zeros((len(model.scores),))  # vocab size

        for w in sentences:
            print(w)
            lattice.set_sentence(w)
            lattice.populate_nodes(model)
            piece_scores = model.scores
            piece_lens = model.piece_lens
            freq = 1  # Assume sentences are unique
            Z = lattice.populate_marginal(freq, piece_scores, expected)
            # print('Z', Z)

            path_best_nodes = lattice.viterbi_iter(piece_scores, piece_lens, model)

            print('|'.join([model.pieces[lattice.node2piece[n]]
                            for n in path_best_nodes]))

            ntokens += len(path_best_nodes)

            assert not np.isnan(Z), \
                "likelihood is NAN. Input sentence may be too long"
            obj -= Z / all_sentence_freq
            print()

        assert not np.isnan(obj)
        print(f'obj: {obj}')

        return expected

    def run_m_step(self, model, expected):
        num_pieces = len(model.pieces)

        assert expected.shape[0] == num_pieces

        new_sentp = []
        sum_ = 0.0
        for i in range(num_pieces):
            freq = expected[i]

            # Filter infrequent sentencepieces here.
            kExpectedFrequencyThreshold = 0.5
            if freq < kExpectedFrequencyThreshold and len(model.pieces[i]) > 1:
                continue

            if freq == 0:
                freq = 1
                print(f'WARNING: This should not happen: single-char piece never assigned: #{model.pieces[i]}#')

            new_sentp.append([model.pieces[i], freq])
            sum_ += freq

        # Here we do not use the original EM, but use the
        # Bayesianified/DPified EM algorithm.
        # https://cs.stanford.edu/~pliang/papers/tutorial-acl2007-talk.pdf
        # This modification will act as a sparse prior.
        logsum = digamma(sum_)

        for i in range(len(new_sentp)):
            new_freq = digamma(new_sentp[i][1]) - logsum
            new_sentp[i][1] = new_freq

        # XXX Return `objective` ?
        return new_sentp  # expected  # , objective, num_tokens

    def train(self, sentences, trainer_spec):  #, normalizer_spec):

        # model = TrainerModel(trainer_spec, normalizer_spec)

        delimiter = trainer_spec['delimiter']

        print_(f'Starting training with {len(sentences)} sentences')
        print_(f'Collecting seed pieces')

        seed_sentp = self.make_seed_sentence_pieces(
            sentences, trainer_spec['num_seed_sentpieces'], delimiter)

        print_(f'Building trie')
        model = Model()
        model.build_trie(sentences, seed_sentp, with_vocab=False,
                         delimiter=delimiter)

        desired_vocab_size = int(trainer_spec['vocab_size'] * 1.1)

        print_(f"Using {len(sentences)} sentences for EM training", "info")

        while True:
            # Sub-EM iteration.
            for it in range(trainer_spec['num_sub_iterations']):

                expected = self.run_e_step(
                    model, sentences)

                new_pieces = self.run_m_step(model, expected)

                # set sentence pieces
                model.build_trie(sentences, new_pieces, with_vocab=False,
                                 delimiter=delimiter)

                objective = 0.0  # XXX
                num_tokens = 0  # XXX

                print_(f"EM sub_iter={it} size={len(model.pieces)} "
                       f"obj={objective} num_tokens={num_tokens} "
                       f"num_tokens/piece={1.0 * num_tokens / model.get_piece_size()}",
                       "info")

            if model.get_piece_size() <= desired_vocab_size:
                break

            new_pieces = self.prune_sentence_pieces(model)
            # model.set_sentence_pieces(std::move(new_sentencepieces))

        # Finally, adjusts the size of sentencepices to be |vocab_size|.
        final_pieces = self.finalize_sentence_pieces(model)
        self.save()

    def prune_sentence_pieces(self, model):

        # lattice =
        # n = len(pieces)
        # always_keep = np.ones((n,),).astype(np.bool)
        # alternatives = np.zeros((n,))

        # First, segments the current sentencepieces to know
        # how each sentencepiece is resegmented if this sentencepiece is removed
        # from the vocabulary.
        # To do so, we take the second best segmentation of sentencepiece[i].
        # alternatives[i] stores the sequence of second best sentencepieces.

        for i, piece in enumerate(pieces):
            # Can you split a piece into smaller ones?
            best_split = best_split(pieces, piece)

            if best_split is None:
                always_keep[i] = True
                continue
            elif len(best_split) >= 2:
                # There is already a cheaper alternative made of more pieces
                always_keep[i] = False
            else:
                # We have an alternative, but it would raise the cost
                assert len(best_split) == 1
                alternatives[i].append(best_split)

        # Second, segments all sentences to compute likelihood
        # with a unigram language model. inverted[i] stores
        # the set of sentence index where the sentencepieces[i] appears.
        vsum = 0.0
        freq = np.zeros((n,))
        inverted = np.zeros((n,))

        for sidx, s in enumerate(sentences):

            vsums = np.zeros((n_threads,))
            freqs
            inverteds
            # std::vector<std::vector<float>> freqs(trainer_spec_.num_threads());
            # std::vector<std::vector<std::vector<int>>> inverteds(
            #     trainer_spec_.num_threads());

            lattice = Lattice()
            lattice.set_sentence(s)
            model.populate_nodes(lattice)

            vsum += 1  # w.second
            for node in lattice.beg_nodes:
                if node > 0:  # XXX why?
                    piece_id = node2piece[node]
                    freqs[piece_id] += 1  # w.second
                    inverted[piece_id].append(piece_id)

        logsum = np.log(freq.sum())
        candidates = []  # pairs (int, float)
        new_pieces = []

        # Finally, computes how likely the LM likelihood is reduced if
        # the sentpieces[i] is removed from the vocabulary.
        # Since the exact computation of loss is difficult, we compute the
        # loss approximately by assuming that all sentencepiece[i] in the sentences
        # are replaced with alternatives[i] when sentencepiece[i] is removed.

        for i, piece in enumerate(model.pieces):

            if freq[i] == 0 or not always_keep[i]:
                # not found in Viterbi path. Can remove this entry safely.
                continue
            elif len(alternatives[i]) == 0:
                # no alternatives. Keeps this entry.
                new_pieces.append(piece)

            else:
                F = 0.0  # the frequency of sentencepieces[i].
                for sidx in inverted[i]:
                    F += 1  # sentences_[n].second
                F /= vsum  # normalizes by all sentence frequency.

                # The logprob with the sentencepiece[i].
                logprob_sp = np.log(freq[i]) - logsum

                #  After removing the sentencepiece[i], its frequency freq[i] is
                #  re-assigned to alternatives.
                #  new_sum = current_sum - freq[i] + freq[i] * alternatives.size()
                #          = current_sum + freq[i] (alternatives - 1)
                logsum_alt = np.log(sum_ + freq[i] * (len(alternatives) - 1))

                # The frequencies of altenatives are increased by freq[i].
                logprob_alt = 0.0
                for j in alternatives[i]:
                    logprob_alt += np.log(freq[n] + freq[i]) - logsum_alt

                # loss: the diff of likelihood after removing the sentencepieces[i].
                loss = F * (logprob_sp - logprob_alt)
                candidates.append(i, loss)

        shrinking_factor = 0.75
        pruned_size = max(desired_vocab_size_, shrinking_factor * len(pieces))

        # Keeps trainer_spec_.shrinking_factor * sentencepieces.size() pieces.
        # shrinking_factor is 0.75 by default.
        for cand in candidates:
            if len(new_pieces) == pruned_size:
                break
            new_pieces.append(cand)

        return new_pieces


if __name__ == '__main__':
    sentences = [line.strip() for line in open('botchan_small.txt')]

    trainer = Trainer()
    spec = {
        'delimiter': '^',  # not in botchan.txt
        'num_seed_sentpieces': 200,
        'vocab_size': 100,
        'num_sub_iterations': 10,
    }
    trainer.train(sentences, spec)

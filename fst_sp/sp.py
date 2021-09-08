from collections import Counter
from esa import ESA
import kaldi_fst_sp
import numpy as np

print_ = print


def to_log_prob(pieces):
    Z = np.log(sum(score for p, score in pieces))
    print(sorted([s for p,s in pieces]))
    pieces = [(p, np.log(score) - Z) for p, score in pieces]
    return pieces


class Trainer:

    def make_seed_sentence_pieces(self, sentences, num_seed_sentpieces,
                                  delimiter='#'):

        corpus = sentences

        print_("Extracting frequent sub strings...")

        # Makes an enhanced suffix array to extract all sub strings occurring
        # more than 2 times in the sentence.
        esa = ESA()
        esa.fit(corpus, delimiter=delimiter, max_piece_len=4)

        seed_sentp = sorted(esa.pieces(), key=lambda p_score: -p_score[1])

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
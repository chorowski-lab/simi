from collections import Counter

import numpy as np


class ESA:

    def fit(self, corpus, delimiter='$', max_piece_len=10**6):
        """
        Args
        ----
        corpus: list of strings

        """

        # if max_piece_len is not None:
        #     # XXX
        #     raise NotImplementedError

        self.trie = None  # Invalidate old trie

        self.corpus = corpus
        # n = sum(len(txt) for txt in corpus)
        n = sum(len(s) for s in corpus)
        self.n = n

        self.suf_counts = Counter()
        for s in corpus:
            self.suf_counts.update([s[idx:idx+max_piece_len] for idx in range(len(s))])

        self.suf = sorted(self.suf_counts.keys())

        n = len(self.suf)
        self.n = n

        # self.suf = np.asarray(sorted(range(n), key=lambda idx: corpus[idx:]))

        print()
        print('Suffix array')
        print('------------')
        print('  ', self.suf)

        for suf in self.suf:
            print('  ', suf)
        print()

        self.lcp = np.zeros((n,), dtype=np.int32)

        for i in range(1, n):
            x = self.suf[i - 1]
            y = self.suf[i]
            for a, b in zip(x, y):
                if a != b or a == delimiter or b == delimiter:
                    break
                self.lcp[i] += 1

        print()
        print('LCP Table')
        print('---------')
        print('  ', self.lcp)
        print()

    def pieces(self):

        n = self.n

        LCP, L, R = 0, 1, 2

        def report(tupl):
            lcp, l, r = tupl
            print(corpus[self.suf[l]:self.suf[l]+int(lcp)], '#\t', tupl)

        def prepare(tupl):
            lcp, l, r = tupl
            piece = self.suf[l][:lcp]
            coverage = sum(self.suf_counts[self.suf[i]] for i in range(l, r + 1)) * len(piece)
            # coverage = (r - l + 1) * len(piece)
            return piece, coverage

        # bottom-up traversal
        stack = [[0, 0, None]]
        for i in range(0, n):  # XXX n + 1 ?
            l = i - 1
            while self.lcp[i] < stack[-1][LCP]:
                stack[-1][R] = i - 1
                interval = stack.pop()
                # report(interval)
                yield prepare(interval)
                l = interval[L]

            if self.lcp[i] > stack[-1][LCP]:
                stack.append([self.lcp[i], l, None])
        stack[-1][R] = n - 1  # XXX n ?
        interval = stack.pop()
        # report(interval)

        # Skip empty '' piece
        if interval[0] != interval[1] != 0:
            yield prepare(interval)

    def insort(self, corpus, suftab, q, beg=0, end=None, left=True):
        """Bisective search that lazily converts suftab to prefixes

        Args
        ----
        corpus (str): string to extract prefixes from
        suftab (iterable): a list of starting indices of sorted suffixes of corpus
        q (str): a query string to be inserted
        left (bool): mimmick `bisect.bisect_left`; if False then mimmick
            `bisect.bisect_right`
        """

        if end is None:
            end = len(suftab) - 1

        if beg == end:
            return beg

        if left:
            pivot = beg + (end - beg) // 2
            val = corpus[suftab[pivot]:][:len(q)]
            # print(beg, pivot, end, val)

            if val < q:
                return self.insort(corpus, suftab, q, beg=pivot+1, end=end)
            else:
                return self.insort(corpus, suftab, q, beg=beg, end=pivot)

        else:
            pivot = beg + (end - beg + 1) // 2
            val = corpus[suftab[pivot]:][:len(q)]
            # print(beg, pivot, end, val)

            if val <= q:
                return self.insort(corpus, suftab, q, beg=pivot, end=end,
                                   left=False)
            else:
                return self.insort(corpus, suftab, q, beg=beg, end=pivot-1,
                                   left=False)

    def search(self, q):
        idx_l = self.insort(self.corpus, self.suf, q)
        idx_r = self.insort(self.corpus, self.suf, q, left=False)
        if self.corpus[self.suf[idx_l]:][:len(q)] == q:
            return (idx_l, idx_r)
        else:
            return (None, None)


if __name__ == '__main__':
    esa = ESA()
    s = 'acaaacatat'
    s = 'acaaacatatdd#dddatat'
    print()
    print(s)
    print()
    esa.fit(s)
    for p, c in esa.pieces():
        print(p, c)

    esa.build_trie()

    print()
    t = 'acatcattattacaacat'
    print(t)
    print()

    for i in range(len(t)):
        for p in esa.prefixes(t[i:]):
            print(f'{" "*i}{p}')

    import sys
    sys.exit(0)

    # esa.fit('Ala ma kota')
    for q in ['ac', 'at', 'aa', 't', 'ta', 'a', 'h']:
        print()
        beg, end = esa.search(q)
        print(q, esa.search(q))
        if beg is not None:
            for i in range(beg, end + 1):
                print(esa.corpus[esa.suf[i]:])

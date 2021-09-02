import os
from pathlib import Path
from argparse import ArgumentParser

from collections import defaultdict

from simi.vectorization import vectorize
from simi.clusterization import cluster_kmeans


def parseArgs():
    parser = ArgumentParser()

    parser.add_argument('segmentation', type=Path,
                        help='Path to the segmentation')
    parser.add_argument('output', type=Path,
                        help='Output path')
    parser.add_argument('--segmentation_format', type=str, default='csv',
                        help='Segmentation format, either \'csv\' (default) or \'txt\'')
    parser.add_argument('--vocab_size', type=int, default=100,
                        help='Size of the output vocab size, 100 by default')
    parser.add_argument('--word2vec_size', type=int, default=100,
                        help='Size of the word2vec vectors, 100 by default')
    parser.add_argument('--word2vec_path', type=Path,
                        help='Path to the word2vec model, if not specified/empty then it will be computed')
    parser.add_argument('--kmeans_path', type=str,
                        help='Path to the kmeans model, if not specified/empty then it will be computed')
    parser.add_argument('--seed', type=int, default=290956,
                        help='Random seed')
    parser.add_argument('--eval', action='store_true', 
                        help='Eval only mode (crash if w2v model is not computed)')
    return parser.parse_args()


class Segmentation(object):
    def __init__(self) -> None:
        self.data = defaultdict(list)
        self.vocab = set()
        super().__init__()

    def __getitem__(self, fname):
        return self.data[fname]

    def to_sentences(self):
        raise NotImplementedError()
    
    def rename(self, word_map):
        raise NotImplementedError()
    
    def save(self, path, filename=None):
        raise NotImplementedError()


class SegmentationCsv(Segmentation):
    def __init__(self, path) -> None:
        super().__init__()
        for csv in Path(path).rglob('*.csv'):
            for line in open(csv, 'r', encoding='utf8'):
                t1, t2, q, kind = line.strip().split(',')
                self.data[csv.stem].append((t1, t2, q, kind))
                self.vocab.add(q)

    def to_sentences(self):
        return list(list(q for _, _, q, _ in d) for d in self.data.values())

    def rename(self, word_map):
        for _, sample in self.data.items():
            for i in range(len(sample)):
                sample[i] = (sample[i][0], sample[i][1], word_map[sample[i][2]], sample[i][3])

    def save(self, path, filename=None):
        if not os.path.exists(path):
            os.makedirs(path)
        for fname, sample in self.data.items():
            with open(path / (fname+'.csv'), 'w') as output:
                for x in sample:
                    output.write(','.join(map(str, x)) + '\n')


class SegmentationTxt(Segmentation):
    def __init__(self, path) -> None:
        super().__init__()
        for line in open(path / 'segmented_outputs.txt', 'r', encoding='utf8'):
            l = line.strip().split()
            self.data[l[0]] = l[1:]
            self.vocab |= set(l[1:])
    
    def to_sentences(self):
        return list(self.data.values())

    def rename(self, word_map):
        for sample in self.data.values():
            for i in range(len(sample)):
                sample[i] = word_map[sample[i]]

    def save(self, path, filename='clustered_outputs.txt'):
        if not os.path.exists(path):
            os.makedirs(path)
        with open(path / filename, 'w', encoding='utf8') as out:
            for fname, sample in self.data.items():
                out.write(f'{fname} {",".join(map(str, sample))}\n')


def run(args):
    print(f'Loading train segmentation...')
    if args.segmentation_format == 'txt':
        segmentation = SegmentationTxt(args.segmentation)
    else:
        segmentation = SegmentationCsv(args.segmentation)
    print(f'Vocabulary size of the segmentation: {len(segmentation.vocab)}')

    assert len(segmentation.vocab) > args.vocab_size, 'Segmentation vocab size must be greater than the output vocab'

    word2vec_path = f'./tmp/word2vec/s{args.seed}' if args.word2vec_path is None else args.word2vec_path
    sentences = segmentation.to_sentences()
    encodings, weights, reconstruct, build_map = vectorize(sentences, word2vec_path, args.word2vec_size, train=not args.eval)


    kmeans_path = f'./tmp/kmeans/s{args.seed}_cosine' if args.kmeans_path is None else args.kmeans_path
    labels = cluster_kmeans(encodings, weights, kmeans_path, args.vocab_size, cosine=True)

    word_map = build_map(labels)
    segmentation.rename(word_map)
    segmentation.save(args.output, filename=f'clustered_outputs_{args.vocab_size}.txt')
    print('Done!')

    
if __name__ == "__main__":
    args = parseArgs()
    run(args)

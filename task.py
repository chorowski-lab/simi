import pathlib
from argparse import ArgumentParser

from simi.dataset import Dataset
from simi.utils import ensure_path
from simi.vectorization import encode_data, train_w2v


def parseArgs():
    parser = ArgumentParser()

    parser.add_argument('trainset', type=pathlib.Path,
                        help='Path to the trainset')
    parser.add_argument('testset', type=pathlib.Path,
                        help='Path to the testset')
    parser.add_argument('word2vec_prefix', type=pathlib.Path,
                        help='')
    parser.add_argument('output', type=pathlib.Path,
                        help='Output path')
    parser.add_argument('--word2vec_size', type=int, default=100,
                        help='Size of the word2vec vectors, 100 by default')
    return parser.parse_args()


class SegmentedOutputs(object):
    def __init__(self, path) -> None:
        super().__init__()
        assert ensure_path(path), f'File: {path} does not exist'
        self.filenames = []
        self.data = []
        self.vocab = set()
        for line in open(path, 'r'):
            l = line.strip().split()
            self.filenames.append(l[0])
            self.data.append(l[1:])
            self.vocab |= set(l[1:])


class Dataset(object):
    def __init__(self, path) -> None:
        super().__init__()
        assert ensure_path(path), f'File: {path} does not exist'
        self.filenames = []
        self.data = []
        for line in open(path, 'r'):
            fname, word = line.strip().split()
            self.filenames.append(fname)
            self.data.append(word)
    

def main(args):
    train_segmentation = SegmentedOutputs(args.trainset)
    test_segmentation = Dataset(args.testset)

    model = train_w2v(train_segmentation.data, args.word2vec_prefix, args.word2vec_size)
    encodings = encode_data(model, test_segmentation.data)

    for fname, vec in zip(test_segmentation.filenames, encodings):
        with open(args.output / f'{fname}.txt', 'w', encoding='utf8') as out:
            out.write(' '.join(map(str, vec)) + '\n')
            out.write(' '.join(map(str, vec)) + '\n')
    

if __name__ == '__main__':
    args = parseArgs()
    main(args)

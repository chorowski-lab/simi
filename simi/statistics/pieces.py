import pathlib
from argparse import ArgumentParser
from collections import defaultdict

from tqdm import tqdm


def parseArgs():
    parser = ArgumentParser()

    parser.add_argument('segmentation', type=pathlib.Path,
                        help='Path to the segmentation')
    return parser.parse_args()


def print_statistics(d, nl='\t'):
    print(f"{nl}Total different pieces: {len(d.keys())}", \
        f"Total pieces: \t\t{sum(d.values())}", \
        f"Min piece count: \t{min(d.values())}", \
        f"Mean piece count: \t{sum(d.values()) / len(d.keys())}", \
        f"Max piece count: \t{max(d.values())}", sep='\n'+nl)
    

def main(args):
    phones = defaultdict(int)
    words = defaultdict(int)

    csvs = set(pathlib.Path(args.segmentation).rglob('*.csv'))

    for csv in tqdm(csvs):
        for line in open(csv, 'r', encoding='utf8'):
            if len(line.strip().split(',')) != 4:
                print(line, csv)
            t1, t2, q, kind = line.strip().split(',')
            if kind == 'phones':
                phones[q] += 1
            elif kind =='words':
                words[q] += 1
            else:
                raise f'Invalid piece type, expected \'phones\' or \'words\', but got {kind} in file {file}.'

    if len(phones.keys()) > 0:
        print('Statistics for phones:')
        print_statistics(phones)

    if len(words.keys()) > 0:
        print('Statistics for words:')
        print_statistics(words)
    

if __name__ == '__main__':
    args = parseArgs()
    main(args)

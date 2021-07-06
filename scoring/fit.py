import argparse
import csv
import glob
import itertools
import re
from collections import Counter, defaultdict
from pathlib import Path

import jiwer
import numpy as np
import tqdm
from intervaltree import Interval, IntervalTree


# spn - non-speech (coughing, etc.)
arpabet = [
    'sil', 'sp', 'spn', 'AA0', 'AA1', 'AA2', 'AE0', 'AE1', 'AE2', 'AH0', 'AH1',
    'AH2', 'AO0', 'AO1', 'AO2', 'AW0', 'AW1', 'AW2', 'AY0', 'AY1', 'AY2', 'B',
    'CH', 'D', 'DH', 'EH0', 'EH1', 'EH2', 'ER0', 'ER1', 'ER2', 'EY0', 'EY1',
    'EY2', 'F', 'G', 'HH', 'IH0', 'IH1', 'IH2', 'IY0', 'IY1', 'IY2', 'JH', 'K',
    'L', 'M', 'N', 'NG', 'OW0', 'OW1', 'OW2', 'OY0', 'OY1', 'OY2', 'P', 'R',
    'S', 'SH', 'T', 'TH', 'UH0', 'UH1', 'UH2', 'UW0', 'UW1', 'UW2', 'V', 'W',
    'Y', 'Z', 'ZH']

arpabet_simple = ['sil', 'sp', 'spn']
arpabet_simple += sorted(set(re.sub('\d', '', a) for a in arpabet[3:]))


def rle(s, counts=True):
    if counts:
        return [(k, sum(1 for _ in v)) for k, v in itertools.groupby(s)]
    else:
        return [k for k, v in itertools.groupby(s)]


def load_alignments(csv_fpath, kind='phones', simplify_arpabet=False):
    assert kind in ('words',  'phones')

    with open(csv_fpath, 'r') as f:
        reader = csv.reader(f)
        tuples = [(float(l[0]), float(l[1]), l[2]) for l in reader if l[-1] == kind]

    if simplify_arpabet:
        a, b, phones = zip(*tuples)
        tuples = list(zip(a, b, simplify(phones)))

    return tuples


def alignment_to_ids(alignment, phone2id, stride=10, simplify_arpabet=False):
    to_idx = lambda sec: int(sec * 1000 // stride)
    ret = np.zeros((to_idx(alignment[-1][1]),), dtype=np.int32)

    for beg, end, ph in alignment:

        if simplify_arpabet:
            ph = simplify([ph])[0]

        ret[to_idx(beg):to_idx(end)] = phone2id[ph]

    return ret


def simplify(phones):
    phones = [re.sub('\d', '0', p) for p in phones]
    phones = [re.sub('(spn|sil)', 'sp', p) for p in phones]
    return phones


def parse_quantized_outputs(path, format='csv', output='ids'):

    assert output in ('ids', 'sentencepieces')

    if format == 'csv':
        char2id = defaultdict(lambda: len(char2id))
        ret = {}
        for fpath in Path(path).rglob('*.csv'):
            key = fpath.stem
            with open(fpath) as csv:
                if output == 'ids':
                    chars = ''.join(line.split(',')[2] for line in csv)
                    ids = np.asarray([char2id[c] for c in chars])
                elif output == 'sentencepieces':
                    ids = [line.split(',')[2] for line in csv]
                else:
                    raise ValueError

            ret[key] = ids

    elif format == 'single_file':
        assert output == 'ids'
        with open(path) as f:
            ret = {l.split()[0]: np.asarray(l.split()[1].strip().split(',')) for l in f}

    else:
        raise ValueError

    return ret


def score_cpc_quantizations(gt_alignments, quantized_outputs,
                            quantized_format='csv', shift=-1, stride=10,
                            subsample=1, simplify_arpabet=False,
                            per_ignore_short_blocks=0):

    simplify_arpabet = True  # XXX

    csvs = {Path(f).stem: f for f in Path(gt_alignments).rglob('*.csv')}

    cpc = parse_quantized_outputs(quantized_outputs, quantized_format)
    print(f'Got {len(cpc)} quantizations')

    phone2id = {ph: i for i, ph in enumerate(arpabet)}

    fit = defaultdict(lambda: Counter())

    dropped_cl = 0
    all_cl = 0

    data = {}

    for key in tqdm.tqdm(csvs.keys()):
        ali = load_alignments(csvs[key])
        gt_phones = [row[2] for row in ali]
        if simplify_arpabet:
            gt_phones = simplify(gt_phones)

        ali = alignment_to_ids(ali, phone2id, stride=10*subsample, simplify_arpabet=simplify_arpabet)

        cl = np.asarray([int(x) for x in cpc[key]])
        cl = cl[::subsample]

        data[key] = {'gt': gt_phones, 'cluster_ids': cl}

        assert ali.shape[0] <= cl.shape[0], f'{ali.shape} {cl.shape} {key}'

        if shift > 0:
            ali = ali[shift:]
        if shift < 0:
            cl = cl[abs(shift):]

        for i, (ph_id, cl_id_pair) in enumerate(zip(ali, cl[:ali.shape[0]])):
            all_cl += 1
            maybe_tuple = lambda x: tuple(x) if type(x) is list or type(x) is np.ndarray else int(x)

            cl_id_pair = maybe_tuple(cl_id_pair)
            fit[cl_id_pair].update([ph_id])

    cl_matching = 0
    cl_all = 0
    for k, v in fit.items():
        all_ = sum(v.values())
        cl_matching += v.most_common(1)[0][1]
        cl_all += all_

    centroid2phone = [None for _ in range(50)]
    for c, cnt in fit.items():
        phones = [arpabet[pair[0]] for pair in cnt.most_common(2)]
        centroid2phone[c] = phones[0]

    collapse = lambda s: [s[0]] + [b for a, b in zip(s[:-1], s[1:]) if a != b]

    # rle = lambda s: [(k, sum(1 for _ in v)) for k, v in itertools.groupby(s)]

    # Go over the data once again and calculate PER
    i = 0
    for key, values in data.items():
        i += 2
        if i < 2:
            continue

        gt = values['gt']
        cl = values['cluster_ids']

        for per_ignore_short_blocks in range(5):
            print('ignore', per_ignore_short_blocks)

            cl = values['cluster_ids']

            # ignore short blocks?
            if per_ignore_short_blocks > 0:
                rle_blocks = [p for p in rle(cl) if p[1] > per_ignore_short_blocks]
                cl = [p[0] for p in rle_blocks]

            # collapse repeated ids: 2 2 2 2 1 1 1 --> 2 1
            cl = collapse(cl)
            # translate to phones and collapse (same phones might get different centroids)
            cl = collapse([centroid2phone[c] for c in cl])
            print(gt)
            print(cl)

            from Bio import pairwise2
            from Bio.pairwise2 import format_alignment

            alignments = pairwise2.align.globalxx(gt, cl, gap_char=[' ']) #' '.join(gt), ' '.join(cl))

            fmt = format_alignment(*alignments[0]).split('\n')
            max_line = 100
            for i in range(0, len(fmt[0]), max_line):
                for l in fmt:
                    print(l[i:i+max_line])
                print()

            print(jiwer.wer(gt, cl))
            print()
        asdasdasd()

    if cl_all == 0:
        print(cl_all, cl_matching, gt_alignments, quantized_outputs)

    return cl_matching / cl_all * 100.0


def score_cpc_quantizations_matching_sentpieces_with_phones(
    gt_alignments, quantized_outputs, quantized_format='csv', shift=-1,
    stride=10, subsample=1, simplify_arpabet=False,
    per_ignore_short_blocks=1, print_sample=0, save_collapsed_phones=False):

    csvs = {Path(f).stem: f for f in Path(gt_alignments).rglob('*.csv')}

    csvs_sp = {Path(f).stem: f for f in Path(quantized_outputs).rglob('*.csv')}

    cpc = parse_quantized_outputs(quantized_outputs, quantized_format,
                                  output='sentencepieces')
    print(f'Got {len(cpc)} quantizations')

    # Treat sentpieces like cluster ids
    id2piece = set()
    for pieces in cpc.values():
        id2piece.update(pieces)
    id2piece = sorted(id2piece)
    piece2id = {p: i for i, p in enumerate(id2piece)}

    cpc = {k: [piece2id[p] for p in v] for k, v in cpc.items()}

    phone2id = {ph: i for i, ph in enumerate(arpabet)}

    fit = defaultdict(lambda: Counter())

    dropped_cl = 0
    all_cl = 0

    data = {}

    piece2phone = defaultdict(lambda: defaultdict(float))

    simplify_arpabet = True  # XXX

    for key in tqdm.tqdm(csvs.keys()):

        ali = load_alignments(csvs[key], simplify_arpabet=simplify_arpabet)
        t = IntervalTree()
        for (start, end, ph) in ali:
            t[start:end] = ph

        sps = load_alignments(csvs_sp[key], simplify_arpabet=False)
        for (start, end, sp) in sps:
            for iv in t[start:end]:
                piece2phone[sp][iv.data] += iv.end - iv.begin

        data[key] = {'gt': [tupl[2] for tupl in ali],
                     'sp': [tupl[2] for tupl in sps]}

    # Now we have piece <-> phone mapping
    piece2phone_ = {}
    for piece, phone_ivs in piece2phone.items():
        piece2phone_[piece] = max(phone_ivs.items(), key=lambda pair: pair[1])[0]
    piece2phone = piece2phone_

    def rle(s, counts=True):
        if counts:
            return [(k, sum(1 for _ in v)) for k, v in itertools.groupby(s)]
        else:
            return [k for k, v in itertools.groupby(s)]

    if save_collapsed_phones:
        # Collapses phones to their regexps, e.g., aaaaaFFFFggg --> a+F+g+
        pieceRLE2phone = defaultdict(Counter)
        with open('pieces2phones.txt', 'w') as f:
            for pi, ph in piece2phone.items():
                rle_ = ''.join(rle(pi, counts=False))
                f.write(f'{pi} {rle_} {ph}\n')
                pieceRLE2phone[rle_][ph] += 1

        s = 0
        for rle_, cnt in pieceRLE2phone.items():
            print(cnt, cnt.most_common(1))
            s += cnt.most_common(1)[0][1]
        print('After collapsing:', len(pieceRLE2phone))
        print('Collapsed properly:', s)

    wers = []
    # Map sentencepieces to phones
    for idx, (key, gt_sp) in enumerate(data.items()):

        gt = gt_sp['gt']
        try:
            sp = [piece2phone[p] for p in gt_sp['sp']]
        except:
            print('Err processing', key)
        sp, counts = zip(*rle(sp, counts=True))

        # Find duration of each sentpiece
        lens = np.asarray([len(p) for p in gt_sp['sp']])
        counts = np.asarray(counts)
        durs = [ar.sum() for ar in np.split(lens, np.cumsum(counts)[:-1])]

        cl = list(sp)

        # drop short ones?
        if per_ignore_short_blocks > 0:
            cl = [cl[i] for i in range(len(cl)) if durs[i] > per_ignore_short_blocks]
            durs = [d for d in durs if d > per_ignore_short_blocks]

        if print_sample is not None and print_sample == idx:

            from Bio import pairwise2
            from Bio.pairwise2 import format_alignment

            alignments = pairwise2.align.globalxx(gt, cl, gap_char=[' '])

            fmt = format_alignment(*alignments[0]).split('\n')
            fmt = fmt[:-2]  # drop last summary rows

            fmt_phones = (re.split('([a-zA-Z0-1]+\s+)', fmt[-1]))
            pos = 0
            for i in range(len(fmt_phones)):
                if fmt_phones[i].strip() != '':
                    fmt_phones[i] = ' ' * len(fmt_phones[i])
                    d = str(durs[pos])
                    pos += 1
                    fmt_phones[i] = d + ' '*(len(fmt_phones[i])-len(d))
            fmt.append(''.join(fmt_phones))

            max_line = 100
            for i in range(0, len(fmt[0]), max_line):
                for l in fmt:
                    print(l[i:i+max_line])
                print()

            print('Pseudophone counts')
            print(counts)
            print('Pseudophone durations')
            print(durs)

        wers.append(jiwer.wer(gt, cl))

    return 100.0 * np.mean(wers)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--dataset", type=str,
                        help="Path to LibriSpeech subset (e.g., dev-clean)")
    parser.add_argument("--alignments", type=str,
                        help="Path to alignments subset, which match the dataset")
    parser.add_argument("--w2v2_clusterings", type=str, default=None,
                        help="Path to clusterings")
    parser.add_argument("--shift", type=int, default=0,
                        help='Shift cluster IDs by N frames (> 0 to right, < 0 to left')
    parser.add_argument("--cpc_subsample", type=int, default=1)
    parser.add_argument("--cpc_clusterings", type=str, default=None)
    parser.add_argument("--cpc_ignore_short_blocks", type=int, default=None,
                        help="Ignore single tokens, e.g., '3,3,3,1,2,2' --> '3,3,3,2,2'")
    parser.add_argument("--cpc_smooth_out", type=int, default=0,
                        help="Apply a heuristic that converts a single id in between other agreeing, e.g., '2,2,1,2,2' --> '2,2,2,2,2'")
    parser.add_argument("--cpc_map_singleton_to_prev", action='store_true',
                        help="Remaps singleton to the prev char, .e.g, '3,3,3,1,2,2' --> '3,3,3,3,2,2'")
    parser.add_argument("--cpc_map_singleton_to_next", action='store_true',
                        help="Remaps singleton to the next char, .e.g, '3,3,3,1,2,2' --> '3,3,3,2,2,2'")
    parser.add_argument("--w2v2_bottom_codebook", action='store_true',
                        help="Use only the bottom codebook")
    parser.add_argument("--w2v2_top_codebook", action='store_true',
                        help="Use only the top codebook")
    args = parser.parse_args()

    csvs = {Path(f).stem: f for f in Path(args.alignments).rglob('*.csv')}
    # wavs = {Path(f).stem: f for f in Path(args.dataset).rglob('*.wav')}

    assert (args.cpc_clusterings is None) != (args.w2v2_clusterings is None)

    if args.cpc_clusterings is not None:
        with open(args.cpc_clusterings, 'r') as f:
            cpc = {l.split()[0]: np.asarray(l.split()[1].strip().split(',')) for l in f}
        stride = 10 * args.cpc_subsample

    if args.w2v2_clusterings is not None:
        clusts = {Path(f).stem: f for f in Path(args.w2v2_clusterings).rglob('*.npy')}
        stride = 20

    phone2id = defaultdict(lambda: len(phone2id))
    phone2id['sil'] = 0

    fit = defaultdict(lambda: Counter())

    dropped_cl = 0
    all_cl = 0


    for key in tqdm.tqdm(csvs.keys()):
        ali = alignment_to_ids(load_alignments(csvs[key]), phone2id, stride=stride)

        # CPC
        if args.cpc_clusterings is not None:

            cl = np.asarray([int(x) for x in cpc[key]])

            if args.cpc_smooth_out >= 1:
                for i in range(cl.shape[0] - 2):
                    if cl[i] == cl[i+2] != cl[i+1]:
                        cl[i+1] = cl[i]

            if args.cpc_smooth_out >= 2:
                for i in range(cl.shape[0] - 3):
                    if cl[i] == cl[i+3] != cl[i+1] and cl[i+1] == cl[i+2]:
                        cl[i+1] = cl[i]
                        cl[i+2] = cl[i]

            if args.cpc_smooth_out >= 3:
                for i in range(cl.shape[0] - 4):
                    if cl[i] == cl[i+4] != cl[i+1] and cl[i+1] == cl[i+2] == cl[i+3]:
                        cl[i+1] = cl[i]
                        cl[i+2] = cl[i]
                        cl[i+3] = cl[i]

            if args.cpc_smooth_out >= 4:
                for i in range(cl.shape[0] - 5):
                    if cl[i] == cl[i+5] != cl[i+1] and cl[i+1] == cl[i+2] == cl[i+3] == cl[i+4]:
                        cl[i+1] = cl[i]
                        cl[i+2] = cl[i]
                        cl[i+3] = cl[i]
                        cl[i+4] = cl[i]

            assert args.cpc_smooth_out <= 4

            if args.cpc_map_singleton_to_prev:
                for i in range(cl.shape[0] - 2):
                    if cl[i] != cl[i+1] != cl[i+2]:
                        cl[i+1] = cl[i]
                for i in range(cl.shape[0] - 2):
                    if cl[i] != cl[i+1] != cl[i+2]:
                        cl[i+1] = cl[i]

            if args.cpc_map_singleton_to_next:
                for i in range(cl.shape[0] - 2):
                    if cl[i] != cl[i+1] != cl[i+2]:
                        cl[i+1] = cl[i+2]
                for i in range(cl.shape[0] - 2):
                    if cl[i] != cl[i+1] != cl[i+2]:
                        cl[i+1] = cl[i+2]

            if args.cpc_ignore_short_blocks:

                l = cl.shape[0]
                b = args.cpc_ignore_short_blocks
                cl_ = np.concatenate([cl, -np.ones((b,), dtype=np.int32)])
                cl_ = np.vstack([cl_[i:l+i] for i in range(b)])
                mask = (cl_ == cl_[0]).sum(axis=0) == b

                # subsample to match GT
                cl = cl[::args.cpc_subsample]
                mask = mask[::args.cpc_subsample]
                assert cl.shape == mask.shape

            else:
                cl = cl[::args.cpc_subsample]

        # w2v2
        if args.w2v2_clusterings is not None:
            cl = np.load(clusts[key])

            if args.w2v2_top_codebook:
                assert cl.shape[1] == 2
                cl = cl[:, 1:]

            if args.w2v2_bottom_codebook:
                assert cl.shape[1] == 2
                cl = cl[:, :1]

            assert not (args.w2v2_top_codebook and args.w2v2_bottom_codebook)

        assert ali.shape[0] <= cl.shape[0], f'{ali.shape} {cl.shape} {key}'

        if args.shift > 0:
            ali = ali[args.shift:]
        if args.shift < 0:
            cl = cl[abs(args.shift):]

        for i, (ph_id, cl_id_pair) in enumerate(zip(ali, cl[:ali.shape[0]])):

            all_cl += 1

            maybe_tuple = lambda x: tuple(x) if type(x) is list or type(x) is np.ndarray else int(x)

            cl_id_pair = maybe_tuple(cl_id_pair)

            if args.cpc_ignore_short_blocks and not mask[i]:
                dropped_cl += 1
            else:
                fit[cl_id_pair].update([ph_id])

    cl_matching = 0
    cl_all = 0
    for k, v in fit.items():

        all_ = sum(v.values())
        # if all_ < 100:
        #     continue

        cl_matching += v.most_common(1)[0][1]
        cl_all += all_

    print(f'{cl_matching / cl_all * 100.0:.2f}%', f'({len(fit)})')
    print(f'all_cl: {all_cl}', f'dropped {dropped_cl} ({dropped_cl / all_cl * 100.0:.2f}%)' if dropped_cl else '')

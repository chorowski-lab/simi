import argparse
import csv
import glob
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import tqdm


def load_alignments(csv_fpath, kind='phones'):
    assert kind in ('words',  'phones')

    with open(csv_fpath, 'r') as f:
        reader = csv.reader(f)
        return [(float(l[0]), float(l[1]), l[2]) for l in reader if l[-1] == kind]


def alignment_to_ids(alignment, phone2id, stride=10):
    to_idx = lambda sec: int(sec * 1000 // stride)
    ret = np.zeros((to_idx(alignment[-1][1]),), dtype=np.int32)

    for beg, end, ph in alignment:
        ret[to_idx(beg):to_idx(end)] = phone2id[ph]

    return ret


def parse_quantized_outputs(path, format='csv'):

    if format == 'csv':
        char2id = defaultdict(lambda: len(char2id))
        ret = {}
        for fpath in Path(path).rglob('*.csv'):
            key = fpath.stem
            with open(fpath) as csv:
                chars = ''.join(line.split(',')[2] for line in csv)
                ids = np.asarray([char2id[c] for c in chars])
            ret[key] = ids

    elif format == 'single_file':
        with open(path) as f:
            ret = {l.split()[0]: np.asarray(l.split()[1].strip().split(',')) for l in f}

    else:
        raise ValueError
        
    return ret


def score_cpc_quantizations(gt_alignments, quantized_outputs, quantized_format='csv', shift=-1, stride=10, subsample=1):

    csvs = {Path(f).stem: f for f in Path(gt_alignments).rglob('*.csv')}

    cpc = parse_quantized_outputs(quantized_outputs, quantized_format)
    print(f'Got {len(cpc)} quantizations')

    phone2id = defaultdict(lambda: len(phone2id))
    phone2id['sil'] = 0
    
    fit = defaultdict(lambda: Counter())
    
    dropped_cl = 0
    all_cl = 0
    
    for key in csvs.keys():
        ali = alignment_to_ids(load_alignments(csvs[key]), phone2id, stride=10*subsample)
        cl = np.asarray([int(x) for x in cpc[key]])
        cl = cl[::subsample]
    
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

    if cl_all == 0:
        print(cl_all, cl_matching, gt_alignments, quantized_outputs)

    return cl_matching / cl_all * 100.0


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--dataset", type=str, help="Path to LibriSpeech subset (e.g., dev-clean)")
    parser.add_argument("--alignments", type=str, help="Path to alignments subset, which match the dataset")
    parser.add_argument("--w2v2_clusterings", type=str, default=None, help="Path to clusterings")
    parser.add_argument("--shift", type=int, default=0, help='Shift cluster IDs by N frames (> 0 to right, < 0 to left')
    parser.add_argument("--cpc_subsample", type=int, default=1)
    parser.add_argument("--cpc_clusterings", type=str, default=None)
    parser.add_argument("--cpc_ignore_short_blocks", type=int, default=None, help="Ignore single tokens, e.g., '3,3,3,1,2,2' --> '3,3,3,2,2'")
    parser.add_argument("--cpc_smooth_out", type=int, default=0, help="Apply a heuristic that converts a single id in between other agreeing, e.g., '2,2,1,2,2' --> '2,2,2,2,2'")
    parser.add_argument("--cpc_map_singleton_to_prev", action='store_true', help="Remaps singleton to the prev char, .e.g, '3,3,3,1,2,2' --> '3,3,3,3,2,2'")
    parser.add_argument("--cpc_map_singleton_to_next", action='store_true', help="Remaps singleton to the next char, .e.g, '3,3,3,1,2,2' --> '3,3,3,2,2,2'")
    parser.add_argument("--w2v2_bottom_codebook", action='store_true', help="Use only the bottom codebook")
    parser.add_argument("--w2v2_top_codebook", action='store_true', help="Use only the top codebook")
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

import os
import pathlib
from functools import reduce
import numpy as np

import pandas

from .score import Score


def format_pipeline(pipeline_desc: str):
    p = pipeline_desc.split('-')

    if len(p) % 2 != 1:
        raise ValueError("Invalid pipeline format, number of steps must be odd")

    return list(int(x) for x in p)

def get_spaces_pos(line):
    pos = []
    i, k, n = 0, 0, len(line)
    while i + k < n:
        if line[i+k] == ' ':
            k += 1
            pos.append(i)
        else:
            i += 1
    return pos


def get_segmentation(sentence):
    pos, i = [], 0
    for word in sentence:
        pos.append(len(word)+i)
        i += len(word)
    return pos[:-1]


def ensure_path(path):
    folderpath = (pathlib.Path(path) / '..').resolve()
    if not os.path.exists(folderpath):
        os.makedirs(folderpath)
    return os.path.exists(path)


int_to_char = "qwertyuioplkjhgfdsazxcvbnmQWERTYUIOPASDFGHJKLZXCVBNM1234567890ąężśźćńłóĄĘĆŚŻŹŃÓŁ!@#$%^&*(),./;'[]\-=<>?:}{|_+`~"
char_to_int = {int_to_char[i] : i for i in range(len(int_to_char))}

def int_array_to_string(array):
    return ''.join(int_to_char[x] for x in array)


def string_to_int_array(s):
    return list(char_to_int[c] for c in s)


def data_to_string_arrays(data):
    for s in data:
        if isinstance(s, np.ndarray) or isinstance(s[0], int) or s[0].dtype == 'int32':
            return list(map(int_array_to_string, data))
        else:
            return data


def save_results(config, results, mode):
    if mode == 'iter1':
        keys = ['seed', 'dataset', 'sp_vocab_size_1']
    elif mode == 'iter2':
        keys = config.keys()
    
    if len(results) == 3:
        data = {**{k : config[k] for k in keys}, 'precision': results[0], 'recall': results[1], 'f_score': results[2]}
    elif len(results) == 5:
        data = {**{k : config[k] for k in keys}, 'precision': results[0], 'recall': results[1], 'f_score': results[2], 'OS': results[3], 'R': results[4]}
    else:
        raise ValueError("Invalid results' length")
    path = f'./results/{mode}.csv'
    if not os.path.exists(path):
        df = pandas.DataFrame([data])
        df.to_csv(path, index=False)
    else:
        df = pandas.read_csv(path)
        if reduce(lambda a, b: a & b, list(df[x] == config[x] for x in keys)).any():
            return
        df = df.append(data, ignore_index=True)
        df.to_csv(path, index=False)


def save_pipeline_results(config: dict, pipeline: str, words_score: Score, phones_score: Score, viterbi=False):
    path = f'./results/simi{"_v" if viterbi else ""}.csv'
    config_keys = ['seed', 'dataset', 'word2vec_size', 'squash']
    data = {**{k: config[k] for k in config_keys}, 
        'pipeline': pipeline,
        'words_PRC': words_score.PRC, 'words_RCL': words_score.RCL, 'words_F': words_score.F,
        'words_HR': words_score.HR, 'words_OS': words_score.OS, 'words_R': words_score.R,
        'phones_PRC': phones_score.PRC, 'phones_RCL': phones_score.RCL, 'phones_F': phones_score.F,
        'phones_HR': phones_score.HR, 'phones_OS': phones_score.OS, 'phones_R': phones_score.R
    }
    if not os.path.exists(path):
        df = pandas.DataFrame([data])
        df.to_csv(path, index=False)
    else:
        df = pandas.read_csv(path)
        if reduce(lambda a, b: a & b, list(df[x] == config[x] for x in config_keys) + [df['pipeline'] == pipeline]).any():
            return
        df = df.append(data, ignore_index=True)
        df.to_csv(path, index=False)

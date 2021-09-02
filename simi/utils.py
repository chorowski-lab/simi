import os
import pathlib

import numpy as np

from simi.stringify import int_array_to_string


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


def data_to_string_arrays(data):
    for s in data:
        if isinstance(s, np.ndarray) or isinstance(s[0], int) or s[0].dtype == 'int32':
            return list(map(int_array_to_string, data))
        else:
            return data

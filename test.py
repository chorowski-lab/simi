import random

import numpy as np
import sentencepiece
from more_itertools import grouper
from argparse import ArgumentParser
from env import *
from src import *

def test():
    data = LibriSpeech('train-clean-100')

    fname, sample = data[0]

    clustering = data.get_clustering(fname)
    a = np.exp(-clustering)
    b = np.sum(a, axis=-1)
    c = np.log(b)
    d = -clustering - c[:,np.newaxis]
    e = np.sum(np.exp(d), axis=-1)
    quantization = np.argmin(clustering, axis=-1)

    # q = np.sum(np.exp(-clustering), axis=-1)

    pass

test()
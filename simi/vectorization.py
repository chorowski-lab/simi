from .utils import ensure_path
from gensim.models import Word2Vec
import numpy as np
import editdistance
from progressbar import ProgressBar
from tqdm import tqdm

def encode_and_format(segmentation, w2v):
    d = dict()
    q = 0
    res = np.zeros(w2v.vectors.shape)
    cnt = np.ones((w2v.vectors.shape[0]), dtype='int32')
    for sentence in segmentation:
        for word in sentence:
            if word in d.keys():
                cnt[d[word]] += 1
            else:
                d[word] = q
                res[q, :] = w2v[word]
                q += 1

    def r(labels):
        nonlocal d, segmentation
        return [[labels[d[word]] for word in sentence] for sentence in segmentation]

    def m(labels):
        nonlocal d
        return { word: labels[d[word]] for word in d.keys() }

    return res, cnt, r, m


def find_closest_encodings(segmentation, w2v):
    words_list = set(w for sentence in segmentation for w in sentence)
    dictionary = list(w2v.vocab)
    d = {}
    q = 0

    res = np.zeros((len(words_list), w2v.vectors.shape[1]))
    cnt = np.ones((len(words_list)), dtype='int32')
    print("Evaluating semantic vectors...", flush=True)
    bar = ProgressBar(maxval=len(words_list))
    bar.start()

    for i, word in enumerate(words_list):
        bar.update(i)
        if word in d.keys():
            cnt[d[word]] += 1
        else:
            ds = [(editdistance.eval(word, w), w) for w in dictionary]
            m, _ = min(ds)
            candidates = [w for (d0, w) in ds if d0 == m]
            v = np.zeros(w2v.vectors.shape[1])
            
            for c in candidates:
                v += w2v[c]
            
            v /= len(candidates)
            d[word] = q
            res[q, :] = v
            q += 1
    bar.finish()

    def r(labels):
        nonlocal d, segmentation
        return [[labels[d[word]] for word in sentence] for sentence in segmentation]

    def m(labels):
        nonlocal d
        return { word: labels[d[word]] for word in d.keys() }

    return res, cnt, r, m


def vectorize(data, path, size, train=True):
    """Run word2vec on the given data.

    Params:

    data: list of sentences (lists of words/strings)
    path: location of the w2v model
    size: size of the model
    """
    if not ensure_path(path):
        if not train:
            raise Exception(f"Tried to eval word2vec, but there is no model at {path}. Maybe set train=True?")
        # train word2vec
        print("Training word2vec model...", flush=True)
        Word2Vec(sentences=data, min_count=1, size=size).save(str(path))
    
    model = Word2Vec.load(str(path))
    if train:
        return encode_and_format(data, model.wv)
    else:
        return find_closest_encodings(data, model.wv)


def train_w2v(data, path, size, train=True):
    """Train word2vec on the given data or, if possible, load existing model.

    Params:

    data: list of sentences (lists of words/strings)
    path: location of the w2v model
    size: size of the model
    """
    if not ensure_path(path):
        if not train:
            raise Exception(f"Tried to eval word2vec, but there is no model at {path}. Maybe set train=True?")
        # train word2vec
        print("Training word2vec model...", flush=True)
        Word2Vec(sentences=data, min_count=1, size=size).save(str(path))
    
    return Word2Vec.load(str(path))


def encode_data(model, data):
    def _encode_word(w2v, word):
        if word in w2v:
            return w2v[word]
        else:
            # if not present in dictionary, we compute the semantic vector
            ds = [(editdistance.eval(word, w), w) for w in w2v.vocab]
            m, _ = min(ds)
            candidates = [w for (d0, w) in ds if d0 == m]
            v = np.zeros(w2v.vectors.shape[1])
            for c in candidates:
                v += w2v[c]
            
            return v / len(candidates)
    
    return [_encode_word(model.wv, w) for w in tqdm(data)]
    
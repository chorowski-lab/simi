# %% IMPORTS & DECLARATIONS
from functools import reduce
from numpy.lib.function_base import select
from sentencepiece import SentencePieceTrainer, SentencePieceProcessor
import os
from gensim.models import Word2Vec
from progressbar import ProgressBar
import random
import numpy as np
import sentencepiece
from sklearn.cluster import KMeans
import pathlib
import pandas


ROOTPATH = 'D:\\zerospeech2021'
TRANSCRIPTIONS_DIR = 'D:\\zerospeech2021\\LibriSpeech-Transcriptions'

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

def flatten(encodings):
    m = sum(len(sentence) for sentence in encodings)
    n = len(encodings[0][0])
    
    res = np.zeros((m, n))
    i = 0
    for sentence in encodings:
        for word in sentence:
            res[i, :] = word
            i += 1
    return res

def ensure_path(path):
    folderpath = (pathlib.Path(path) / '..').resolve()
    if not os.path.exists(folderpath):
        os.makedirs(folderpath)
    return os.path.exists(path)

int_to_char_data = "qwertyuioplkjhgfdsazxcvbnmQWERTYUIOPASDFGHJKLZXCVBNM1234567890!@#$%^&*(),./;'[]\-=<>?:}{|_+`~ążźćęńłó"
def int_array_to_string(array):
    return ''.join(int_to_char_data[x] for x in array)

def save_results(config, results, mode):
    if mode == 'iter1':
        keys = ['seed', 'dataset', 'sp_vocab_size_1']
    elif mode == 'iter2':
        keys = config.keys()
    
    data = {**{k : config[k] for k in keys}, 'precision': results[0], 'recall': results[1], 'f_score': results[2]}
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


# %% DATASET

class Dataset:
    def __init__(self, name):
        self.dict = set()
        self.gt_seg = []
        self.data = []
        self.lengths = []
        for line in open(os.path.join(TRANSCRIPTIONS_DIR, name + '.txt'), 'r', encoding='utf8'):
            line = line.strip()
            self.gt_seg.append(get_spaces_pos(line))
            self.dict |= set(line)
            line = line.replace(' ', '')
            self.data.append(line)
            self.lengths.append(len(line))
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]
        
    def f1score(self, segmentation):
        tp, fp, fn = 0, 0, 0
        bar = ProgressBar(max_value=len(self))
        bar.start()
        for i in range(len(self)):
            bar.update(i)
            # q = len(line_gt.replace(' ', ''))
            gt_pos = set(self.gt_seg[i])
            seg_pos = set(segmentation[i])
            _tp = len(gt_pos & seg_pos)
            _fp = len(seg_pos - gt_pos)
            _fn = len(gt_pos - seg_pos)
            tp += _tp
            fp += _fp
            fn += _fn
            # tn += q - _tp - _fp - _fn
        bar.finish()
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 / (1 / recall + 1 / precision)
        print('### Precision: {:.2f}%'.format(precision * 100))
        print('### Recall:    {:.2f}%'.format(recall * 100))
        print('### F1:        {:.2f}%'.format(f1 * 100))

        return precision, recall, f1



# %% MAIN PROCESS DEFINITION

def run(config, data):
    vocab_size = config['sp_vocab_size_1']
    SP_PREFIX_PATH = f'./models/sentencepiece/{dataset}/vocab_size_{vocab_size}/s{seed}'
    SP_MODEL_PATH = SP_PREFIX_PATH + '.model'
    
    if not ensure_path(SP_MODEL_PATH):
        # train sentencepiece
        print("Training sentencepiece model...")
        SentencePieceTrainer.train(sentence_iterator=iter(data.data), model_prefix=SP_PREFIX_PATH, vocab_size=vocab_size)
    
    # encode
    sp = SentencePieceProcessor()
    sp.Load(SP_MODEL_PATH)
    sp_encodings = sp.Encode(data.data, out_type=str)
    sp_formatted = list(list(word.replace('▁', '').strip() for word in encoding if word.replace('▁', '').strip() != '') for encoding in sp_encodings)
    
    for i in range(len(data)):
        assert ''.join(sp_formatted[i]) == data[i]

    sp_segmentation = list(get_spaces_pos(' '.join(sentence)) for sentence in sp_formatted)
    
    word2vec_size = config['word2vec_size']
    W2V_PATH = f'./models/word2vec/{dataset}/vocab_size_{vocab_size}/w2v_size_{word2vec_size}/s{seed}.model'

    if not ensure_path(W2V_PATH):
        print("Training word2vec model...")
        Word2Vec(sentences=sp_formatted, min_count=1, size=word2vec_size).save(W2V_PATH)
    
    model = Word2Vec.load(W2V_PATH)

    w2v_encodings = flatten([
        [model.wv[word] for word in sentence]
        for sentence in sp_formatted])

    n_clusters = config['kmeans_n_clusters']
    KMEANS_LABELS_PATH = f'./models/kmeans/{dataset}/vocab_size_{vocab_size}/w2v_size_{word2vec_size}/n_clusters_{n_clusters}/s{seed}_labels.npy'
    KMEANS_CENTERS_PATH = f'./models/kmeans/{dataset}/vocab_size_{vocab_size}/w2v_size_{word2vec_size}/n_clusters_{n_clusters}/s{seed}_centers.npy'

    if not ensure_path(KMEANS_LABELS_PATH):
        print("Running kmeans...")
        kmeans = KMeans(n_clusters=n_clusters).fit(w2v_encodings)
        np.save(KMEANS_CENTERS_PATH, kmeans.cluster_centers_, allow_pickle=True)
        np.save(KMEANS_LABELS_PATH, kmeans.labels_, allow_pickle=True)
    
    labels = np.load(KMEANS_LABELS_PATH, allow_pickle=True)
    
    reformatted = []
    i = 0
    for sentence in sp_formatted:
        reformatted.append(int_array_to_string(labels[i:i+len(sentence)]))
        i += len(sentence)

    after_vocab_size = config['sp_vocab_size_2']
    SP2_PREFIX_PATH = f'./models/sentencepiece_after/{dataset}/vocab_size_{vocab_size}/w2v_size_{word2vec_size}/n_clusters_{n_clusters}/after_vocab_size_{after_vocab_size}/s{seed}'
    SP2_MODEL_PATH = SP2_PREFIX_PATH + '.model'
    if not ensure_path(SP2_MODEL_PATH):
        # train sentencepiece
        print("Training second sentencepiece model...")
        SentencePieceTrainer.train(sentence_iterator=iter(reformatted), model_prefix=SP2_PREFIX_PATH, vocab_size=after_vocab_size)
    
    # encode
    sp2 = SentencePieceProcessor()
    sp2.Load(SP2_MODEL_PATH)
    sp2_encodings = sp2.Encode(reformatted, out_type=str)
    sp2_formatted = list(list(word.replace('▁', '').strip() for word in encoding if word.replace('▁', '').strip() != '') for encoding in sp2_encodings)
    sp2_segmentation_internal = list(get_spaces_pos(' '.join(sentence)) for sentence in sp2_formatted)
    sp2_segmentation = []
    for sp1, sp2 in zip(sp_segmentation, sp2_segmentation_internal):
        sp2_segmentation.append(np.array(sp1)[np.array(sp2, dtype='int32')-1])

    r1 = data.f1score(sp_segmentation)
    save_results(config, r1, 'iter1')
    r2 = data.f1score(sp2_segmentation)
    save_results(config, r2, 'iter2')
    # sp_formatted = list(list(word.replace('▁', '').strip() for word in encoding if word.replace('▁', '').strip() != '') for encoding in sp_encodings)

# %% RUN

config = {
    'seed': 290956,
    'dataset': 'm1/train-clean-100',
    'sp_vocab_size_1': 20000,
    'sp_vocab_size_2': 5000,
    'word2vec_size': 100,
    'kmeans_n_clusters': 50
}

seed = config['seed']
random.seed(seed)
np.random.seed(seed)
sentencepiece.set_random_generator_seed(seed)

dataset = config['dataset']
data = Dataset(dataset)

run(config, data)

# %% TRAIN SENTENCEPIECE
vocab_size = 3000
SentencePieceTrainer.train(sentence_iterator=iter(data.data), model_prefix=f'./models/sentencepiece/{dataset}_{vocab_size}', vocab_size=vocab_size)

# %% ENCODE WITH SENTENCEPIECE
sp = SentencePieceProcessor()
sp.Load(f'./models/sentencepiece/{dataset}_{vocab_size}.model')
encodings = sp.Encode(data.data, out_type=str)
formatted = list(' '.join(encoding).replace('▁', '').strip() for encoding in encodings)

# %% SCORE SENTENCEPIECE ENCODINGS
data.f1score(formatted)
print(formatted[0])
print(formatted[1])
print(formatted[182])

# %% TRAIN WORD2VEC

model = Word2Vec(sentences=encodings)
model.save(f'./models/word2vec/{dataset}.model')

# %% LOAD WORD2VEC

model = Word2Vec.load(f'./models/word2vec/{dataset}_{vocab_size}.model')

# %% TEST WORD2VEC

print(model.wv.most_similar('england'))

# %% RUN WORD2VEC ON GROUD-TRUTH

sentences = list(map(lambda line: line.split(), data.gt))
vocab_size = 1000

model = Word2Vec(sentences=sentences)

# model.save(f'./models/word2vec/{dataset}_{vocab_size}_GT_TEST.model')
model.wv.most_similar('hollow')
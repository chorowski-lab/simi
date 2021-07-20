import os
from progressbar import ProgressBar
from env import TRANSCRIPTIONS_DIR, ROOTPATH
from .utils import get_spaces_pos
from .segmentation import LibriSpeechSegmentation
import numpy as np
from progressbar import ProgressBar


class Data(object):
    def __init__(self, path):
        self.filenames = []
        self.data = []
        self.clusterings = None
        for line in open(path, 'r', encoding='utf8'):
            fname, data = line.strip().split()
            self.filenames.append(fname)
            self.data.append([int(x) for x in data.split(',')])
            
    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.filenames[i], self.data[i]

    def load_clusterings(self, path, alpha=1.0):
        self.clusterings = Clusterings(path, self.filenames, alpha=alpha)


class Clusterings(object):
    def __init__(self, path, id_to_filename, alpha=1.0):
        self.path = path
        self.filenames = id_to_filename
        self.alpha = alpha
        self.cache = dict()
    
    def __getitem__(self, i):
        if i not in self.cache.keys():
            dists = np.load(os.path.join(self.path, self.filenames[i] + '.npy')).reshape(-1, 50)
            logprobs = np.log(1.0 / dists) * self.alpha
            self.cache[i] = logprobs - np.log(np.sum(np.exp(logprobs), axis=-1))[:,np.newaxis]

        return self.cache[i]

    def __len__(self):
        return len(self.filenames)


class LibriSpeech(Data):
    def __init__(self, name, path=None):
        if name.startswith('LibriSpeech/'):
            name = name[12:]
        if path is None:
            path = f'/pio/data/zerospeech2021/quantized/LibriSpeech/{name}/quantized_outputs.txt'
        super(LibriSpeech, self).__init__(path)
        self.name = name
        self.clusterings = Clusterings(f'/pio/scratch/1/i290956/zs2021/clusterings/LibriSpeech/{name}', self.filenames)
        self.segmentation = None
    
    # def get_clustering_distribution(self, filename):
    #     dists = np.load(os.path.join(self.clusterings_path, filename + '.npy')).reshape(-1, 50)
    #     return - dists - np.log(np.sum(np.exp(-dists), axis=-1))[:,np.newaxis]

    def rate_segmentation(self, segmentation, tolerance=2):
        if self.segmentation is None:
            self.segmentation = LibriSpeechSegmentation(self.name)
        return self.segmentation.rate(segmentation, self.filenames, tolerance)
    
    def squash(self):
        segmentation = []
        for i in range(len(self.data)):
            sample = [self.data[i][0]]
            segmentation.append([])
            for j in range(1, len(self.data[i])):
                if sample[-1] != self.data[i][j]:
                    sample.append(self.data[i][j])
                    segmentation[-1].append(j)
            self.data[i] = sample
        return segmentation


class Dataset:
    def __init__(self, name):
        self.dict = set()
        self.gt_seg = []
        self.data = []
        self.lengths = []
        self.name = name
        for line in open(os.path.join(ROOTPATH, TRANSCRIPTIONS_DIR, name + '.txt'), 'r', encoding='utf8'):
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


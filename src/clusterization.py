from .utils import ensure_path
import numpy as np
import pickle
from sklearn.cluster import KMeans

def cluster_kmeans(data, weights, path, n_clusters, train=True):
    labels_path = path + '_labels.npy'
    centers_path = path + '_centers.npy'
    labels = None

    if not ensure_path(path):
        if not train:
            raise Exception(f"Tried to cluster data, but there is no kmeans model at {path}. Maybe set train=True?")
        # run k-means
        print("Running kmeans...", flush=True)
        kmeans = KMeans(n_clusters=n_clusters).fit(data, sample_weight=weights)
        pickle.dump(kmeans, open(path, "wb"))
        # np.save(centers_path, kmeans.cluster_centers_, allow_pickle=True)
        # np.save(labels_path, kmeans.labels_, allow_pickle=True)
        labels = kmeans.labels_
    else:
        kmeans = pickle.load(open(path, "rb"))
        # labels = np.load(labels_path, allow_pickle=True)
        # centers = np.load(centers_path, allow_pickle=True)
        labels = kmeans.predict(data)
    return labels
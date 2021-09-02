from simi.utils import ensure_path
import numpy as np
import pickle
from sklearn.cluster import KMeans

def cluster_kmeans(data, weights, path, n_clusters, train=True, cosine=False):
    labels = None

    if cosine:
        length = np.sqrt((data**2).sum(axis=1))[:,None]
        length[length <= 0] = 0.00000001 # sometimes these vectors might be zero and following line would crash
        data = data / length

    if not ensure_path(path):
        if not train:
            raise Exception(f"Tried to cluster data, but there is no kmeans model at {path}. Maybe set train=True?")
        # run k-means
        print("Running kmeans...", flush=True)
        kmeans = KMeans(n_clusters=n_clusters).fit(data, sample_weight=weights)
        pickle.dump(kmeans, open(path, "wb"))
        labels = kmeans.labels_
    else:
        kmeans = pickle.load(open(path, "rb"))
        labels = kmeans.predict(data)
        
    return labels
from simi.utils import ensure_path
import numpy as np
import pickle
from sklearn.cluster import KMeans
import faiss

def cluster_kmeans(data, weights, path, n_clusters, train=True, cosine=False):
    labels = None

    if cosine:
        length = np.sqrt((data**2).sum(axis=1))[:, None]
        # sometimes these vectors might be zero and following line would crash
        length[length <= 0] = 0.00000001
        data = data / length

    if not ensure_path(path):
        if not train:
            raise Exception(
                f"Tried to cluster data, but there is no kmeans model at {path}. Maybe set train=True?")
        # run k-means
        print("Training kmeans...", flush=True)
        kmeans = faiss.Kmeans(data.shape[-1], n_clusters, verbose=True)
        kmeans.train(data.astype(np.float32), weights=weights.astype(np.float32))
        print("Computing labels...", flush=True)
        _, labels = kmeans.assign(data.astype(np.float32))
        pickle.dump(kmeans.centroids, open(path, "wb"))
    else:
        centroids = pickle.load(open(path, "rb"))
        print("Computing labels...", flush=True)
        labels = np.argmax(np.dot(data, centroids.T), axis=-1)

    return labels

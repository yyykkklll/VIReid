import numpy as np
from sklearn.cluster import KMeans
import torch

def run_kmeans(features, num_clusters):
    """
    features: numpy array [N, dim]
    """
    print(f"Running K-Means with K={num_clusters}...")
    kmeans = KMeans(n_clusters=num_clusters, n_init=10, random_state=42)
    kmeans.fit(features)
    return kmeans.labels_, kmeans.cluster_centers_
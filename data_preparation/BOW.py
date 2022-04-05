from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

import numpy as np

class BOW:

    def __init__(self, bag_of_descriptors, N):
        self.N = N
        self.kmeans = KMeans(n_clusters=N)
        self.kmeans.fit(bag_of_descriptors)
        self.features_dict = self.kmeans.cluster_centers_

    def predict(self, descriptors):
        features = []
        for desc in descriptors:
            feature = [0] * self.N
            dist = cdist(desc, self.features_dict)
            labels = np.argmin(dist, axis=1)
            for l in labels:
                feature[l] += 1
            features.append(feature)
        features = np.array(features)
        return features

from sklearn.cluster import MiniBatchKMeans
from scipy.spatial.distance import cdist

import numpy as np

class BOW:

    def __init__(self, bag_of_descriptors, N):
        self.N = N
        self.kmeans = MiniBatchKMeans(n_clusters=N)
        self.kmeans.fit(bag_of_descriptors)

    def predict(self, descriptors):
        return [np.histogram(self.kmeans.predict(desc), bins=range(self.kmeans.n_clusters), density=True)[0] for desc in descriptors]

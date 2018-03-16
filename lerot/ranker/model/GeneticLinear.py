import numpy as np

from lerot.utils import sample_unit_sphere


class GeneticLinear:

    def __init__(self, feature_count):
        self.feature_count = feature_count

    def get_feature_count(self):
        return self.feature_count

    def initialize_weights(self, method):
        if method == "zero":
            return np.zeros(self.feature_count)
        elif method == "random":
            return sample_unit_sphere(self.feature_count) * 0.01

    def score(self, features, w):
        return np.dot(features, w)

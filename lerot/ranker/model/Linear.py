import numpy as np

from lerot.ranker.model.AbstractRankingModel import AbstractRankingModel


class Linear(AbstractRankingModel):

    def score(self, features, w):
        return np.dot(features, w)

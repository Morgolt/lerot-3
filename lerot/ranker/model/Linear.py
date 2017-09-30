from lerot.ranker.model.AbstractRankingModel import AbstractRankingModel
import numpy as np

class Linear(AbstractRankingModel):

    def score(self, features, w):
        return np.dot(features, w)
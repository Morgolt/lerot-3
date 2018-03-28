from lerot.interleave.AbstractInterleavedComparison import AbstractInterleavedComparison
from lerot.query import Query
import numpy as np


class ProbabilisticMultileave(AbstractInterleavedComparison):
    def __init__(self) -> None:
        super().__init__()
        self.tau = 3.0

    def interleave(self, rankers: list, query: Query, length: int) -> list:
        rankings = []
        for r in rankers:
            r.init_ranking(query)
            rankings.append(r.docids)
        self.nrrankers = len(rankers)
        k = min(min([len(r) for r in rankings]), length)

        inverted_rankings =

        assignments = np.random.randint(0, self.nrrankers, k)

        denom = np.zeros(k) + np.sum(1 / (np.arange(self.nrankers) + 1) ** self.tau)
        probs = 1. /


    def infer_outcome(self, l, a, c, query):
        pass

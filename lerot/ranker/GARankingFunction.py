import numpy as np

from lerot.ranker.AbstractRankingFunction import AbstractRankingFunction
from lerot.ranker.model import Linear
from lerot.utils import rank


class GARankingFunction(AbstractRankingFunction):
    def __init__(self, w):
        self.w = np.array(w)
        self.ranking_model = Linear(len(w))
        self.ties = 'random'

    def init_ranking(self, query):
        self.qid = query.get_qid()
        scores = self.ranking_model.score(query.get_feature_vectors(), self.w)
        ranks = rank(scores, ties='random', reverse=False)
        ranked_docids = []
        for pos, docid in enumerate(query.__docids__):
            ranked_docids.append((ranks[pos], docid))
        # sort docids by rank
        ranked_docids.sort(reverse=True)
        self.docids = [docid for (_, docid) in ranked_docids]
        # break ties randomly and sort ranks to compute probabilities
        ranks = np.asarray([i + 1.0 for i in sorted(rank(scores, ties=self.ties, reverse=False))])
        # determine probabilities based on (reverse) document ranks
        max_rank = len(ranks)
        tmp_val = max_rank / pow(ranks, 3.0)
        self.probs = tmp_val / sum(tmp_val)

    def get_ranking(self):
        return self.docids

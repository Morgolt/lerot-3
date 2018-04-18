import numpy as np

from lerot.ranker.AbstractRankingFunction import AbstractRankingFunction
from lerot.ranker.model import Linear
from lerot.utils import rank


class GARankingFunction(AbstractRankingFunction):
    def document_count(self):
        return len(self.docids)

    def __init__(self, w):
        self.w = np.array(w)
        self.ranking_model = Linear(len(w))
        self.ties = 'random'

    def init_ranking(self, query):
        self.qid = query.get_qid()
        scores = self.ranking_model.score(query.get_feature_vectors(), self.w)
        self.ranks = rank(scores, reverse=False)
        ranked_docids = [(self.ranks[pos], docid) for pos, docid in enumerate(query.__docids__)]
        # sort docids by rank
        ranked_docids.sort(reverse=True)
        self.docids = [docid for (_, docid) in ranked_docids]


    def get_ranking(self):
        return self.docids

    def next(self):
        """produce the next document by random sampling, or
        deterministically"""

        # if there are no more documents
        if len(self.docids) < 1:
            raise Exception("There are no more documents to be selected")

        # if there's only one document
        if len(self.docids) == 1:
            self.probs = np.delete(self.probs, 0)  # should be empty now
            pick = self.docids.pop()  # pop, because it's a list
            return pick

        # sample if there are more documents
        # how to do this efficiently?
        # take cumulative probabilities, then do binary search?
        # if we sort docs and probabilities, we can start search at the
        # beginning. This will be efficient, because we'll look at the most
        # likely docs first.
        cumprobs = np.cumsum(self.probs)
        pick = -1
        rand = np.random.rand()  # produces a float in range [0.0, 1.0)
        for pos, cp in enumerate(cumprobs):
            if rand < cp:
                pick = self.docids.pop(pos)  # pop, because it's a list
                break

        if pick == -1:
            print("Cumprobs:", cumprobs)
            print("rand", rand)
            raise Exception("Could not select document!")
        # renormalize
        self.probs = np.delete(self.probs, pos)  # delete, it's a numpy array
        self.probs = self.probs / sum(self.probs)
        return pick

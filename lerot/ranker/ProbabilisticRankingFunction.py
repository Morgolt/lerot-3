import numpy as np

from lerot.ranker.AbstractRankingFunction import AbstractRankingFunction
from lerot.utils import rank


class ProbabilisticRankingFunction(AbstractRankingFunction):

    def init_ranking(self, query):
        self.dirty = False
        self.qid = query.get_qid()
        scores = self.ranking_model.score(query.get_feature_vectors(), self.w.transpose())
        # rank scores
        self.ranks = rank(scores, ties=self.ties, reverse=False)
        # get docids for the ranked scores
        ranked_docids = []
        for pos, docid in enumerate(query.__docids__):
            ranked_docids.append((self.ranks[pos], docid))
        # sort docids by rank
        ranked_docids.sort(reverse=True)
        self.docids = [docid for (_, docid) in ranked_docids]
        # break ties randomly and sort ranks to compute probabilities
        self.ranks = np.asarray([i + 1.0 for i in
                            sorted(rank(scores, ties=self.ties, reverse=False))])
        # determine probabilities based on (reverse) document ranks
        max_rank = len(self.ranks)
        tmp_val = max_rank / pow(self.ranks, self.ranker_type)
        self.probs = tmp_val / sum(tmp_val)

    def document_count(self):
        return len(self.docids)

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

    def next_det(self):
        pos = 0  # first is the most likely document
        pick = self.docids.pop(pos)
        # renormalize
        self.probs = np.delete(self.probs, pos)  # delete, it's a numpy array
        self.probs = self.probs / sum(self.probs)
        return pick

    def next_random(self):
        """produce a random next document"""

        # if there are no more documents
        if len(self.docids) < 1:
            raise Exception("There are no more documents to be selected")
        # otherwise, return a random document
        rn = np.random.randint(0, len(self.docids) - 1)
        return self.docids.pop(rn)

    def get_ranking(self):
        return self.docids

    def get_document_probability(self, docid):
        """get probability of producing doc as the next document drawn"""
        pos = self.docids.index(docid)
        return self.probs[pos]

    def rm_document(self, docid):
        """remove doc from list of available docs and adjust probabilities"""
        # find position of the document
        try:
            pos = self.docids.index(docid)
        except ValueError:
            #            raise Exception("Cannot remove %s. Current document list: %s "
            #                            "for qid: %s. \nProbably, you are trying to "
            #                            "interleave two identical rankers." %
            #                            (docid, self.docids, self.qid))
            return
        # delete doc and renormalize
        self.docids.pop(pos)
        self.probs = np.delete(self.probs, pos)
        self.probs = self.probs / sum(self.probs)

    def getDocs(self, numdocs=None):
        """ Copied from StatelessRankingFunction. """
        if numdocs is None:
            return self.docids
        else:
            return self.docids[:numdocs]

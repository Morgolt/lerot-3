from typing import Tuple

import numpy as np

from lerot.interleave.AbstractInterleavedComparison import AbstractInterleavedComparison
from lerot.query import Query


def preferences_of_list(probs, n_samples):
    """
    ARGS:
    -probs: clicked docs x rankers matrix with probabilities ranker added clicked doc  (use probability_of_list)
    -n_samples: number of samples to base preference matrix on

    RETURNS:
    - preference matrix: matrix (rankers x rankers) in this matrix [x,y] > 0 means x won over y and [x,y] < 0 means x lost from y
      the value is analogous to the (average) degree of preference
    """

    n_clicks = probs.shape[0]
    n_rankers = probs.shape[1]
    upper = np.cumsum(probs, axis=1)
    lower = np.zeros(probs.shape)
    lower[:, 1:] += upper[:, :-1]
    # flip coins, coins fall between lower and upper
    coinflips = np.random.rand(n_clicks, n_samples)
    # make copies for each sample and each ranker
    comps = coinflips[:, :, None]
    # determine where each coin landed
    log_assign = np.logical_and(comps > lower[:, None, :], comps <= upper[:, None, :])
    # click count per ranker (samples x rankers)
    click_count = np.sum(log_assign, axis=0)
    # prefs = np.sign(click_count[:, :, None] - click_count[:, None, :])
    # return np.mean(np.sum(prefs, axis=0) / float(n_samples), axis=1)
    return np.sum(click_count, axis=0) / n_samples


class ProbabilisticMultileave(AbstractInterleavedComparison):
    def __init__(self) -> None:
        super().__init__()
        self.tau = 3.0

    def interleave(self, rankers: list, query: Query, length: int) -> Tuple[list, np.ndarray]:
        rankings = []
        for r in rankers:
            r.init_ranking(query)
            rankings.append(r.ranks)
        self.nrankers = len(rankers)

        inverted_rankings = np.array(rankings)
        self.inverted_rankings = inverted_rankings
        n = inverted_rankings.shape[1]
        k = min(n, length)

        assignments = np.random.randint(0, self.nrankers, k)

        denom = np.zeros(k) + np.sum(1 / (np.arange(n) + 1) ** self.tau)
        probs = 1. / (inverted_rankings[assignments, :] + 1) ** self.tau
        ranking = []
        docids = query.get_docids()

        for i in range(k):
            upper = np.cumsum(probs[i, :])
            lower = np.zeros(upper.shape)
            lower[1:] += upper[:-1]

            coinflip = np.random.rand()

            logic = np.logical_and(lower / denom[i] < coinflip,
                                   upper / denom[i] >= coinflip)

            raw_i = np.where(logic)[0][0]

            ranking.append(docids[raw_i])

            docids[raw_i:-1] = docids[raw_i + 1:]

            denom -= probs[:, raw_i]
            if raw_i < n - 1:
                probs[:, raw_i:-1] = probs[:, raw_i + 1:]
        self.ranking = ranking
        return ranking, self.inverted_rankings

    def infer_outcome(self, l, a, c, query):
        l = np.array(l)
        if np.any(c):
            click_ids = np.nonzero(c)[0]
            probs = self.probability_of_list(l, a, click_ids)
            prefs = preferences_of_list(probs, 10000)
            return prefs

    def probability_of_list(self, result_list, inverted_rankings, clicked_docs, tau=3.0):
        '''
        ARGS: (all np.array of docids)
        - result_list: the multileaved list
        - inverted_rankings: matrix (rankers x documents) where [x,y] corresponds to the rank of doc y in ranker x
        - clicked_docs: the indices of the clicked_docs

        RETURNS
        -sigmas: matrix (rankers x clicked_docs) with probabilty ranker added clicked doc
        '''
        n = inverted_rankings.shape[1]
        # normalization denominator for the complete ranking
        sigmoid_total = np.sum(float(1) / (np.arange(n) + 1) ** tau)
        docids = np.array([doc.get_id() for doc in result_list])
        # cumsum is used to renormalize the probs, it contains the part
        # the denominator that has to be removed (due to previously added docs)
        cumsum = np.zeros(inverted_rankings.shape)

        cumsum[:, docids[1:]] = np.cumsum(
            (float(1) / (inverted_rankings[:, docids[:-1]] + 1.) ** tau),
            axis=1)

        # make sure inverted rankings is of dtype float
        sigmas = 1 / (inverted_rankings[:, clicked_docs].T + 1.) ** tau
        sigmas /= sigmoid_total - cumsum[:, clicked_docs].T

        return (sigmas / np.sum(sigmas, axis=1)[:, None])

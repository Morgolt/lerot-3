import numpy as np

from lerot.interleave.AbstractInterleavedComparison import AbstractInterleavedComparison


class TeamDraftMultileave(AbstractInterleavedComparison):
    """Baseline team draft method."""

    def __init__(self, arg_str=None):
        pass

    def interleave(self, rankers, query, length):
        rankings = []
        for r in rankers:
            r.init_ranking(query)
            rankings.append(r.docids)
        self.nrrankers = len(rankers)
        length = min(min([len(r) for r in rankings]), length)
        # start with empty document list and assignments
        l = []
        lassignments = []
        # determine overlap in top results
        index = 0
        for i in range(length):
            if len(set([r[i] for r in rankings])) == 1:
                l.append(rankings[0][i])
                lassignments.append(-1)
                index += 1
            else:
                break

        indexes = [index] * len(rankings)

        assignments = [0] * len(rankings)

        while len(l) < length:
            minassignment = min(assignments)
            minassigned = [i for i, a in enumerate(assignments) if a == minassignment]

            rindex = np.random.choice(minassigned)
            assignments[rindex] += 1
            lassignments.append(rindex)
            # add first doc that is not in list already
            while True:
                next_doc = rankings[rindex][indexes[rindex]]
                indexes[rindex] += 1
                if next_doc not in l:
                    l.append(next_doc)
                    break

        return np.asarray(l), np.asarray(lassignments)

    def infer_outcome(self, l, a, c, query):
        """assign clicks for contributed documents"""
        creds = []
        for r in range(self.nrrankers):
            credit = sum([1 if val_a == r and val_c == 1 else 0 for val_a, val_c in zip(list(a), list(c))])
            creds.append(credit)
        return creds
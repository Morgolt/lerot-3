import numpy as np

from lerot.evaluation.AbstractEval import AbstractEval


class DcgEval(AbstractEval):
    """Compute DCG (with gain = 2**rel-1 and log2 discount)."""

    def get_dcg(self, ranked_labels, cutoff=-1):
        """
        Get the dcg value of a list ranking.
        Does not check if the numer for ranked labels is smaller than cutoff.
        """
        if (cutoff == -1):
            cutoff = len(ranked_labels)

        rank = np.arange(cutoff)
        return ((np.power(2, np.asarray(ranked_labels[:cutoff])) - 1) /
                np.log2(2 + rank)).sum()

    def evaluate_ranking(self, ranking, query, cutoff=-1):
        """
        Compute DCG for the provided ranking. The ranking is expected
        to contain document ids in rank order.
        """
        if cutoff == -1 or cutoff > len(ranking):
            cutoff = len(ranking)

        # get labels for the sorted docids
        sorted_labels = [0] * cutoff
        for i in range(cutoff):
            sorted_labels[i] = query.get_label(ranking[i])
        return self.get_dcg(sorted_labels, cutoff)

    def get_value(self, ranking, labels, orientations, cutoff=-1):
        """
        Compute the value of the metric
        - ranking contains the list of documents to evaluate
        - labels are the relevance labels for all the documents, even those
          that are not in the ranking; labels[doc.get_id()] is the relevance of
          doc
        - orientations contains orientation values for the verticals;
          orientations[doc.get_type()] is the orientation value for the
          doc (from 0 to 1).
        """
        return self.get_dcg([labels[doc.get_id()] for doc in ranking], cutoff)
from lerot import utils
from lerot.ranker.model.GeneticLinear import GeneticLinear
from lerot.utils import rank


class GeneticRankingFunction:
    def __init__(self, ranker_arg_str, ties, feature_count, init="random", sample="sample_unit_sphere"):
        self.feature_count = feature_count
        self.ranking_model = GeneticLinear(self.feature_count)
        self.sample = utils.sample_unit_sphere
        self.ties = ties
        self.w = self.ranking_model.initialize_weights(init)


    def init_ranking(self, query):
        self.quid = query.get_qid()
        self.ranking_model =
        scores = self.ranking_model.score(query.get_feature_vectors(),
                                          self.w.transpose())
        ranks = rank(scores, reverse=False, ties=self.ties)

        ranked_docids = []
        for pos, docid in enumerate(query.get_docids()):
            ranked_docids.append((ranks[pos], docid))
        ranked_docids.sort(reverse=True)
        self.docids = [docid for (_, docid) in ranked_docids]

    def get_document_probability(self, docid):
        pass

    def rm_document(self, docid):
        pass

    def document_count(self):
        pass

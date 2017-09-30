from lerot.utils import get_class


class AbstractRankingFunction:
    """Abstract base class for ranking functions."""

    def __init__(self,
                 ranker_arg_str,
                 ties,
                 feature_count,
                 init="random",
                 sample="sample_unit_sphere"):

        self.feature_count = feature_count
        ranking_model_str = "ranker.model.Linear"
        for arg in ranker_arg_str:
            if type(arg) is str and arg.startswith("ranker.model"):
                ranking_model_str = arg
            elif arg is not None:
                try:
                    self.ranker_type = float(arg)
                except ValueError:
                    pass
        self.ranking_model = get_class(ranking_model_str)(feature_count)

        if sample:
            self.sample = get_class("utils." + sample)

        self.ties = ties
        self.w = self.ranking_model.initialize_weights(init)

    def score(self, features):
        return self.ranking_model.score(features, self.w.transpose())

    def get_candidate_weight(self, delta):
        """Delta is a change parameter: how much are your weights affected by
        the weight change?"""
        # Some random value from the n-sphere,
        u = self.sample(self.ranking_model.get_feature_count())
        return self.w + delta * u, u

    def init_ranking(self, query):
        self.dirty = False
        raise NotImplementedError("Derived class needs to implement "
            "init_ranking.")

    def next(self):
        self.dirty = True
        raise NotImplementedError("Derived class needs to implement "
            "next.")

    def next_det(self):
        self.dirty = True
        raise NotImplementedError("Derived class needs to implement "
            "next_det.")

    def next_random(self):
        self.dirty = True
        raise NotImplementedError("Derived class needs to implement "
            "next_random.")

    def get_document_probability(self, docid):
        raise NotImplementedError("Derived class needs to implement "
            "get_document_probability.")


    def getDocs(self, numdocs=None):
        if not hasattr(self, "dirty"):
            raise NotImplementedError("Derived class should (re)set self.dirty")
        if self.dirty:
            raise Exception("Always call init_ranking() before getDocs()!")
        docs = []
        i = 0
        while True:
            if numdocs is not None and i >= numdocs:
                break
            try:
                docs.append(self.next())
            except Exception as e:
                break
            i += 1
        return docs

    def rm_document(self, docid):
        raise NotImplementedError("Derived class needs to implement "
            "rm_document.")

    def document_count(self):
        raise NotImplementedError("Derived class needs to implement "
            "document_count.")

    def update_weights(self, w, alpha=None):
        """update weight vector"""
        if alpha is None:
            self.w = w
        else:
            self.w = self.w + alpha * w
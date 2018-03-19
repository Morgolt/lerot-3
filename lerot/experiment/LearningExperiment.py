class LearningExperiment:

    def __init__(self, train, test, feature_count, log_fn, **kwargs):
        self.log_filename = log_fn
        self.train = train
        self.test = test
        self.feature_count = feature_count

        self.num_queries = kwargs['num_queries']
        self.um = kwargs['user_model']


    def run(self):
        qids = sorted(self.train.keys())
        query_length = len(qids)

        for query_count in range(self.num_queries):
            qid = self._sample_qid(qids, query_count, query_length)



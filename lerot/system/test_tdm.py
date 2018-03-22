from pathlib import Path

import numpy as np

from lerot.evaluation.NdcgEval import NdcgEval
from lerot.interleave.TeamDraftMultileave import TeamDraftMultileave
from lerot.query import load_queries
from lerot.ranker.ProbabilisticRankingFunction import ProbabilisticRankingFunction
from lerot.user.CascadeUserModel import CascadeUserModel


class Experiment:
    def __init__(self, folder: Path, um: str) -> None:
        train = str(folder / "train.txt")
        test = str(folder / "test.txt")

        self.n_rankers = 5
        self.num_features = 45
        self.train = load_queries(train, self.num_features)
        self.test = load_queries(test, self.num_features)

        print("Found {} train queries and {} test queries.".format(len(self.train), len(self.test)))

        self.ml = TeamDraftMultileave()
        self.rankers = np.array([ProbabilisticRankingFunction("1", "random", self.num_features)
                                 for _ in range(self.n_rankers)])
        self.eval = NdcgEval()
        self.um = CascadeUserModel(um)
        self.cutoff = 10

        print("OK")

    def run(self, n_impressions):
        results = {
            "offline_ndcg": [],
            "online_ndcg": []
        }
        for i in range(n_impressions):
            print("Impression #", i)
            res = self.impression()
            for eval in results.keys():
                results[eval].append(res[eval])

    def impression(self):
        query = self.train[np.random.choice(self.train.get_qids())]
        td_results = self.impression_tdm(query)
        print(td_results)
        return td_results

    def impression_tdm(self, query):
        ranking, a = self.ml.interleave(self.rankers, query, self.cutoff)
        clicks = self.um.get_clicks(ranking, query.get_labels())
        creds = np.array(self.ml.infer_outcome(ranking, a, clicks, query))
        online_ndcg = self.eval.evaluate_ranking(ranking, query, 10)
        offline_ndcg = self.evaluate_offline(creds)
        return {"online_ndcg": online_ndcg, "offline_ndcg": offline_ndcg}

    def evaluate_offline(self, creds):
        # todo: evaluate only for best ranker
        winners_idx = np.argwhere(creds == np.amax(creds)).flatten()
        winners = self.rankers[winners_idx]
        result = []
        for query in self.test:
            off_ndcg = []
            for ranker in winners:
                ranker.init_ranking(query)
                off_ndcg.append(self.eval.evaluate_ranking(ranker.get_ranking(), query, 10))
            result.append(np.mean(off_ndcg))
        return np.mean(result)


if __name__ == '__main__':
    np.seterr("ignore")
    np.random.seed(42)
    DATA_PATH = Path("D:/Projects/Lerot3/data")
    fold = DATA_PATH / "OHSUMED" / "Fold1"
    experiment = Experiment(fold, "--p_click 0:0, 1:0.5, 2:1.0 --p_stop  0:0, 1:0, 2:0")
    experiment.run(1)

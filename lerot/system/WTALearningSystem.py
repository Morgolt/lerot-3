import argparse
import copy

import numpy as np

from lerot.system.AbstractLearningSystem import AbstractLearningSystem
from lerot.utils import split_arg_str, get_class


class WTALearningSystem(AbstractLearningSystem):
    # todo: genetic ranking
    def __init__(self, feature_count, arg_str):
        # parse arguments
        parser = argparse.ArgumentParser(
            description="Initialize retrieval system with the specified feedback and learning mechanism.",
            prog="WTALearningSystem")
        parser.add_argument("-w", "--init_weights", help="Initialization method for weights (random, zero).",
                            required=True)
        parser.add_argument("--sample_weights", default="sample_unit_sphere")
        parser.add_argument("-c", "--comparison", required=True)
        parser.add_argument("-f", "--comparison_args", nargs="*")
        parser.add_argument("-r", "--ranker", required=True)
        parser.add_argument("-s", "--ranker_args", nargs="*", default=[])
        parser.add_argument("-t", "--ranker_tie", default="random")
        parser.add_argument("-d", "--delta", required=True, type=str)
        parser.add_argument("-a", "--alpha", required=True, type=str)
        parser.add_argument("-nr", "--num_rankers", required=True, type=int)
        parser.add_argument("-up", "--update", required=True, type=str)
        args = vars(parser.parse_known_args(split_arg_str(arg_str))[0])

        self.ranker_class = get_class(args["ranker"])
        self.num_rankers = args['num_rankers']
        self.ranker_args = args["ranker_args"]
        self.ranker_tie = args["ranker_tie"]
        self.sample_weights = args["sample_weights"]
        self.init_weights = args["init_weights"]
        self.feature_count = feature_count
        self.ranker = self.ranker_class(self.ranker_args, self.ranker_tie, self.feature_count,
                                        sample=self.sample_weights, init=self.init_weights)
        self.update = args["update"]
        if "," in args["delta"]:
            self.delta = np.array([float(x) for x in args["delta"].split(",")])
        else:
            self.delta = float(args["delta"])
        if "," in args["alpha"]:
            self.alpha = np.array([float(x) for x in args["alpha"].split(",")])
        else:
            self.alpha = float(args["alpha"])

        self.comparison_class = get_class(args['comparison'])
        if "comparison_args" in args and args["comparison_args"] is not None:
            self.comparison_args = " ".join(args["comparison_args"])
            self.comparison_args = self.comparison_args.strip("\"")
        else:
            self.comparison_args = None
        self.comparison = self.comparison_class(self.comparison_args)
        self.query_count = 0

    def get_ranked_list(self, query):
        """
        Already multileaved
        :param query: qid
        :return: multileaved list for user impression
        """
        self.query_count += 1
        candidates = [self.ranker]
        u = []
        for i in range(self.num_rankers):
            candidate_ranker, candidate_u = self._get_new_candidate()
            candidates.append(candidate_ranker)
            u.append(candidate_u)

        (l, context) = self.comparison.interleave(candidates, query, 10)
        self.current_l = l
        self.current_context = context
        self.current_query = query
        self.current_u = np.array(u)
        self.current_candidates = np.array(candidates)
        return l

    def update_solution(self, clicks):
        creds = self.comparison.infer_outcome(self.current_l,
                                              self.current_context,
                                              clicks,
                                              self.current_query)
        winners_idx = np.argwhere(creds[1:] == np.amax(creds)).flatten()
        if max(creds) != creds[0]:
            target_u = 0
            if self.update == 'wta':
                target_u = self.current_u[np.random.choice(winners_idx)]
            elif self.update == 'mw':
                target_u = np.sum(self.current_u) / len(self.current_u)
            self.ranker.update_weights(target_u, self.alpha)
        return self.get_solution()

    def get_solution(self):
        return self.ranker

    def _get_new_candidate(self):
        w, u = self.ranker.get_candidate_weight(self.delta)
        candidate_ranker = copy.deepcopy(self.ranker)
        candidate_ranker.update_weights(w)
        return candidate_ranker, u

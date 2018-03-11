import argparse

from lerot.retrieval_system import AbstractLearningSystem
from lerot.utils import rank, split_arg_str, get_class


class GeneticListwiseLearningSystem(AbstractLearningSystem):
# todo: implement GA with deap
    def __init__(self, feature_count, arg_str):
        parser = argparse.ArgumentParser(description="Initialize retrieval "
                                                     "system with the specified feedback and learning mechanism.",
                                         prog="GeneticListwiseLearningSystem")
        parser.add_argument("-w", "--init_weights", help="Initialization "
                                                         "method for weights (random, zero).", required=True)
        parser.add_argument("--sample_weights", default="sample_unit_sphere")
        parser.add_argument("-c", "--comparison", required=True)
        parser.add_argument("-f", "--comparison_args", nargs="*")
        parser.add_argument("-r", "--ranker", required=True)
        parser.add_argument("-s", "--ranker_args", nargs="*", default=[])
        parser.add_argument("-t", "--ranker_tie", default="random")
        parser.add_argument("-d", "--delta", required=True, type=str)
        parser.add_argument("-a", "--alpha", required=True, type=str)
        parser.add_argument("--anneal", type=int, default=0)
        parser.add_argument("--normalize", default="False")
        args = vars(parser.parse_known_args(split_arg_str(arg_str))[0])

        self.ranker_class = get_class(args["ranker"])
        self.ranker_args = args["ranker_args"]
        self.ranker_tie = args["ranker_tie"]
        self.sample_weights = args["sample_weights"]
        self.init_weights = args["init_weights"]
        self.feature_count = feature_count
        self.ranker = self.ranker_class(self.ranker_args,
                                        self.ranker_tie,
                                        self.feature_count,
                                        sample=self.sample_weights,
                                        init=self.init_weights)

        if "," in args["delta"]:
            self.delta = np.array([float(x) for x in args["delta"].split(",")])
        else:
            self.delta = float(args["delta"])
        if "," in args["alpha"]:
            self.alpha = np.array([float(x) for x in args["alpha"].split(",")])
        else:
            self.alpha = float(args["alpha"])

        self.anneal = args["anneal"]

        self.comparison_class = get_class(args["comparison"])
        if "comparison_args" in args and args["comparison_args"] is not None:
            self.comparison_args = " ".join(args["comparison_args"])
            self.comparison_args = self.comparison_args.strip("\"")
        else:
            self.comparison_args = None
        self.comparison = self.comparison_class(self.comparison_args)
        self.query_count = 0

    def _get_new_candidate(self):
        pass

    def _get_candidate(self):
        pass

    def get_ranked_list(self, query, get_new_candidate=True):
        self.query_count += 1
        if get_new_candidate:
            self.candidate_ranker, self.current_u = self._get_candidate()

        (l, context) = self.comparison.interleave(self.ranker, self.candidate_ranker, query, 10)
        self.current_l = l
        self.current_context = context
        self.current_query = query
        return l


    def update_solution(self, clicks):
        pass

    def get_solution(self):
        pass
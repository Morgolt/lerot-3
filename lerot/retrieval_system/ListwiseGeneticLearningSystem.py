import argparse

from lerot import utils
from lerot.utils import split_arg_str, get_class
from deap import base
from deap import creator
from deap import tools


class ListwiseGeneticLearningSystem:
    def __init__(self, feature_count, arg_str):
        # parse arguments
        parser = argparse.ArgumentParser(description="Initialize retrieval "
                                                     "system with the specified feedback and learning mechanism.",
                                         prog="ListwiseLearningSystem")
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

        # GA args
        creator.create("FitnessClicks", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessClicks)
        self.toolbox = base.Toolbox()
        self.toolbox.register("sample_unit_sphere", utils.sample_unit_sphere)
        self.toolbox.register("ind", tools.initIterate, creator.Individual, self.toolbox.sample_unit_sphere, n=feature_count)
        self.toolbox.register("pop", tools.initRepeat, list, self.toolbox.ind)
        self.pop = self.toolbox.pop(n=2)
        self.toolbox.evaluate()

        # ranker arguments
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
        # comparison args
        self.comparison_class = get_class(args["comparison"])
        if "comparison_args" in args and args["comparison_args"] is not None:
            self.comparison_args = " ".join(args["comparison_args"])
            self.comparison_args = self.comparison_args.strip("\"")
        else:
            self.comparison_args = None
        self.comparison = self.comparison_class(self.comparison_args)
        # init state
        self.query_count = 0



    def _get_new_candidate(self):
        # Get new candidate by

    def get_solution(self):
        return self.ranker

    def get_individual(self):


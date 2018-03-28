import array

from deap import base, creator, tools

from lerot.system import WTALearningSystem
from lerot.utils import sample_unit_sphere


class GALearningSystem(WTALearningSystem):
    def __init__(self, feature_count, arg_str):
        super().__init__(feature_count, arg_str)

        creator.create("FitnessClicks", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessClicks)

        self.toolbox = base.Toolbox()
        self.toolbox.register("attr_float", sample_unit_sphere, self.feature_count)
        self.toolbox.register("ind", tools.initIterate, creator.Individual, self.toolbox.attr_float)

        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.ind)
        self.pop = self.toolbox.population(n=self.num_rankers)

    def update_solution(self, clicks):
        creds = self.comparison.infer_outcome(self.current_l,
                                              self.current_context,
                                              clicks,
                                              self.current_query)

import logging
import warnings

import sys

from numpy.linalg import norm

from lerot.experiment.AbstractLearningExperiment import AbstractLearningExperiment
from lerot.utils import get_cosine_similarity


class LearningExperiment(AbstractLearningExperiment):
    def run(self):
        qids = sorted(self.train.keys())
        query_length = len(qids)
        online_evaluation = {}
        offline_evaluation = {}

        for eval_name, eval_dict in self.evaluations:
            dict_name = eval_name + '@' + str(eval_dict['cutoff'])

            if dict_name in online_evaluation:
                warnings.warn("Duplicate evaluation arguments, omitting..")
                continue
            online_evaluation[dict_name] = []
            offline_evaluation[dict_name] = []

        similarities = [.0]
        weights = []

        for query_count in range(self.num_queries):
            previous_solution_w = self.system.get_solution().w
            qid = self._sample_qid(qids, query_count, query_length)
            query = self.train[qid]

            result_list = self.system.get_ranked_list(query)

            # online eval
            for eval_name, eval_dict in self.evaluations:
                a = float(eval_dict['eval_class'].evaluate_ranking(result_list, query, eval_dict['cutoff']))
                online_evaluation[eval_name + '@' + str(eval_dict['cutoff'])].append(a)

            clicks = self.um.get_clicks(result_list, query.get_labels())

            current_solution = self.system.update_solution(clicks)

            for eval_name, eval_dict in self.evaluations:
                # Create dict name as done above
                dict_name = eval_name + '@' + str(eval_dict['cutoff'])
                if (not (previous_solution_w == current_solution.w).all()) or len(offline_evaluation[dict_name]) == 0:
                    e1 = eval_dict['eval_class'].evaluate_all(current_solution, self.test, eval_dict['cutoff'])
                    offline_evaluation[dict_name].append(float(e1))
                else:
                    offline_evaluation[dict_name].append(offline_evaluation[dict_name][-1])

            similarities.append(float(get_cosine_similarity(previous_solution_w, current_solution.w)))
            weights.append(','.join([str(w) for w in previous_solution_w]))

        # Print new line for the next run
        logging.info('Done')

        summary = {"weight_sim": similarities, "weights": weights}

        for eval_name, eval_dict in self.evaluations:
            dict_name = eval_name + '@' + str(eval_dict['cutoff'])
            logging.info("Final offline %s = %.3f" % (dict_name, offline_evaluation[dict_name][-1]))
            summary["online_" + dict_name] = online_evaluation[dict_name]
            summary["offline_" + dict_name] = offline_evaluation[dict_name]

        logging.info("Length of final weight vector = %.3f" % norm(current_solution.w))
        return summary





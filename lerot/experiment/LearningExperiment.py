import warnings

import logging

import sys

from numpy.linalg import norm

from lerot.experiment.AbstractLearningExperiment import AbstractLearningExperiment
from lerot.utils import get_cosine_similarity


class LearningExperiment(AbstractLearningExperiment):
    """
    Represents an experiment in which a retrieval system learns from
    implicit user feedback. The experiment is initialized as specified in the
    provided arguments, or config file.
    """

    def run(self):
        """
        A single run of the experiment.
        """
        query_keys = sorted(self.training_queries.keys())
        query_length = len(query_keys)

        online_evaluation = {}
        offline_test_evaluation = {}

        for eval_name, eval_dict in self.evaluations:
            dict_name = eval_name + '@' + str(eval_dict['cutoff'])
            # Stop if there are duplicate evaluations
            if dict_name in online_evaluation:
                warnings.warn("Duplicate evaluation arguments, omitting..")
                continue
            online_evaluation[dict_name] = []
            offline_test_evaluation[dict_name] = []
        similarities = [.0]

        # Process queries
        for query_count in range(self.num_queries):
            logging.debug("Query nr: {}".format(query_count))
            previous_solution_w = self.system.get_solution().w
            qid = self._sample_qid(query_keys, query_count, query_length)
            query = self.training_queries[qid]
            # get result list for the current query from the system
            result_list = self.system.get_ranked_list(query)

            # Online evaluation
            for eval_name, eval_dict in self.evaluations:
                a = float(eval_dict['eval_class'].evaluate_ranking(result_list,
                          query, eval_dict['cutoff']))
                online_evaluation[eval_name + '@' +
                                  str(eval_dict['cutoff'])].append(a)

            # generate click feedback
            clicks = self.um.get_clicks(result_list, query.get_labels())
            # send feedback to system
            current_solution = self.system.update_solution(clicks)

            # compute current offline performance (over all documents)
            for eval_name, eval_dict in self.evaluations:
                # Create dict name as done above
                dict_name = eval_name + '@' + str(eval_dict['cutoff'])
                if (not (previous_solution_w == current_solution.w).all()) or \
                        len(offline_test_evaluation[dict_name]) == 0:
                    e1 = eval_dict['eval_class'].evaluate_all(
                        current_solution, self.test_queries,
                        eval_dict['cutoff'])
                    offline_test_evaluation[dict_name].append(float(e1))
                else:
                    offline_test_evaluation[dict_name].append(
                                    offline_test_evaluation[dict_name][-1])

            similarities.append(float(get_cosine_similarity(
                previous_solution_w, current_solution.w)))

        # Print new line for the next run
        sys.stdout.write('\nDone')
        sys.stdout.write('\n')
        sys.stdout.flush()

        # Finalize evaluation measures after training is done
        summary = {"weight_sim": similarities, "final_weights":
                   previous_solution_w.tolist()}
        for eval_name, eval_dict in self.evaluations:
            dict_name = eval_name + '@' + str(eval_dict['cutoff'])
            logging.info("Final offline %s = %.3f" % (dict_name,
                         offline_test_evaluation[dict_name][-1]))
            summary["online_ndcg" + dict_name] = online_evaluation[dict_name]
            summary["offline_ndcg" + dict_name] = offline_test_evaluation[dict_name]
        logging.info("Length of final weight vector = %.3f" %
                     norm(current_solution.w))
        return summary

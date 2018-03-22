import argparse
import logging
import os
import sys

import numpy as np
import yaml

from lerot.query import load_queries
from lerot.utils import get_class


class GenericExperiment:
    def __init__(self, args_str=None):
        # parse arguments
        parser = argparse.ArgumentParser(description="""
            Construct and run a learning experiment. Provide either the name
            of a config file from which the experiment configuration is
            read, or provide all arguments listed under Command line. If
            both are provided the  config file is ignored.""",
                                         prog=self.__class__.__name__)

        # option 1: use a config file
        file_group = parser.add_argument_group("FILE")
        file_group.add_argument("-f", "--file", help="Filename of the config "
                                                     "file from which the experiment details"
                                                     " should be read.")

        # option 2: specify all experiment details as arguments
        detail_group = parser.add_argument_group("DETAILS")
        detail_group.add_argument("-i", "--training_queries",
                                  help="File from which to load the training queries (svmlight "
                                       "format).")
        detail_group.add_argument("-j", "--test_queries",
                                  help="File from which to load the test queries (svmlight format).")
        detail_group.add_argument("-c", "--feature_count", type=int,
                                  help="The number of features included in the data.")
        detail_group.add_argument("-r", "--num_runs", type=int,
                                  help="Number of runs (how many times to repeat the experiment).")
        detail_group.add_argument("-q", "--num_queries", type=int,
                                  help="Number of queries in each run.")
        detail_group.add_argument("-u", "--user_model",
                                  help="Class implementing a user model.")
        detail_group.add_argument("-v", "--user_model_args",
                                  help="Arguments for initializing the user model.")
        # the retrieval system maintains ranking functions, accepts queries and
        # generates result lists, and in return receives user clicks to learn
        # from
        detail_group.add_argument("-s", "--system",
                                  help="Which system to use (e.g., pairwise, listwise).")
        detail_group.add_argument("-a", "--system_args", help="Arguments for "
                                                              "the system (comparison method, learning "
                                                              "algorithm and parameters...).")
        detail_group.add_argument("-o", "--output_dir",
                                  help="(Empty) directory for storing output generated by this"
                                       " experiment. Subdirectory for different folds will be generated"
                                       "automatically.")
        detail_group.add_argument("--output_dir_overwrite", default="False")
        detail_group.add_argument("-p", "--output_prefix",
                                  help="Prefix to be added to output filenames, e.g., the name of "
                                       "the data set, fold, etc. Output files will be stored as "
                                       "OUTPUT_DIR/PREFIX-RUN_ID.txt")
        detail_group.add_argument("-e", "--experimenter", help="Experimenter type.")
        detail_group.add_argument("-sd", "--seed", type=int)
        # run the parser
        if args_str:
            args = parser.parse_known_args(args_str.split())[0]
        else:
            args = parser.parse_known_args()[0]

        # determine whether to use config file or detailed args
        self.experiment_args = None
        self.args_file = args.file
        if args.file:
            config_file = open(args.file)
            self.experiment_args = yaml.load(config_file)
            config_file.close()
            # overwrite with command-line options if given
            for arg, value in vars(args).items():
                if value:
                    self.experiment_args[arg] = value
        else:
            self.experiment_args = vars(args)

        # workaround - check if we have all the arguments needed
        if not ("training_queries" in self.experiment_args and
                "test_queries" in self.experiment_args and
                "feature_count" in self.experiment_args and
                "num_runs" in self.experiment_args and
                "num_queries" in self.experiment_args and
                "user_model" in self.experiment_args and
                "user_model_args" in self.experiment_args and
                "system" in self.experiment_args and
                "system_args" in self.experiment_args and
                "output_dir" in self.experiment_args):
            parser.print_help()
            sys.exit("Missing required arguments, please check the program"
                     " arguments or configuration file. %s" %
                     self.experiment_args)

        # set default values for optional arguments
        if "query_sampling_method" not in self.experiment_args:
            self.experiment_args["query_sampling_method"] = "random"
        if "output_dir_overwrite" not in self.experiment_args:
            self.experiment_args["output_dir_overwrite"] = False
        if "experimenter" not in self.experiment_args:
            self.experiment_args["experimenter"] = \
                "experiment.LearningExperiment.LearningExperiment"
        if "evaluation" not in self.experiment_args:
            self.experiment_args["evaluation"] = "evaluation.NdcgEval"
        if "processes" not in self.experiment_args:
            self.experiment_args["processes"] = 0
        if "seed" not in self.experiment_args:
            np.random.seed(42)
        else:
            np.random.seed(self.experiment_args['seed'])

        # locate or create directory for the current fold
        if not os.path.exists(self.experiment_args["output_dir"]):
            os.makedirs(self.experiment_args["output_dir"])
        elif not (self.experiment_args["output_dir_overwrite"]) and \
                os.listdir(self.experiment_args["output_dir"]):
            # make sure the output directory is empty
            raise Exception(
                "Output dir %s is not an empty directory. Please"
                " use a different directory, or move contents out of the way."
                % self.experiment_args["output_dir"])

        logging.basicConfig(format='%(levelname)s %(module)s %(asctime)s: %(message)s',
                            level=logging.INFO)
        logging.info("Arguments: %s" % self.experiment_args)

        # Printing out arguments that are used in execution
        for k, v in sorted(self.experiment_args.items()):
            logging.info("\t%s: %s" % (k, v))
        config_bk = os.path.join(self.experiment_args["output_dir"],
                                 "config_bk.yml")
        logging.info("Backing up configuration to: %s" % config_bk)
        with open(config_bk, "w") as config_bk_file:
            yaml.dump(self.experiment_args, config_bk_file, default_flow_style=False)

        # load training and test queries
        training_file = self.experiment_args["training_queries"]
        test_file = self.experiment_args["test_queries"]
        self.feature_count = self.experiment_args["feature_count"]
        logging.info("Loading training data: %s " % training_file)
        self.training_queries = load_queries(training_file, self.feature_count)
        logging.info("... found %d queries." %
                     self.training_queries.get_size())
        logging.info("Loading test data: %s " % test_file)
        self.test_queries = load_queries(test_file, self.feature_count)
        logging.info("... found %d queries." % self.test_queries.get_size())

        # initialize and run the experiment num_run times
        self.num_runs = self.experiment_args["num_runs"]
        self.output_dir = self.experiment_args["output_dir"]
        self.output_prefix = self.experiment_args["output_prefix"]
        self.experimenter = get_class(self.experiment_args["experimenter"])

    def run(self):
        if self.experiment_args["processes"] > 1:
            from multiprocessing import Pool
            pool = Pool(processes=self.experiment_args["processes"])
            results = [
                pool.apply_async(self._run, (run_count,))
                for run_count in range(self.num_runs)
            ]
            pool.close()
            pool.join()
            for result in results:
                logging.info("Ready: {}".format(result.ready()))
                logging.info("Successful: {}".format(result.successful()))
            return [result.get() for result in results]
        else:
            # Run the experiment num_runs times and return the list of results
            return [self._run(run_id) for run_id in range(self.num_runs)]

    def _run(self, run_id):
        logging.info("run %d starts" % run_id)
        aux_log_file = os.path.join(self.output_dir, "_%s-%d.txt" %
                                    (self.output_prefix, run_id))
        with open(aux_log_file, "w") as aux_log_fh:
            # Returns summary after running an experiment
            summarized_experiment = self.run_experiment(aux_log_fh)

        # Setup result log file
        log_file = os.path.join(self.output_dir, "%s-%d.txt" %
                                (self.output_prefix, run_id))
        with open(log_file, "w") as log_fh:
            # todo: switch to csv
            yaml.dump(summarized_experiment, log_fh, default_flow_style=False)
        logging.info("run %d done \n" % run_id)

        return summarized_experiment

    def run_experiment(self, aux_log_fh):
        # Run an experiment with given parameters
        experiment = self.experimenter(self.training_queries, self.test_queries, self.feature_count, aux_log_fh,
                                       self.experiment_args)

        return experiment.run()
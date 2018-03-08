import argparse
import glob
import gzip

import logging
import os
import random

import numpy
import yaml

from lerot.experiment.GenericExpriment import GenericExperiment
from lerot.utils import get_class


class MetaExperiment:
    def __init__(self):
        # parse arguments
        parser = argparse.ArgumentParser(description="""Meta experiment""")

        file_group = parser.add_argument_group("FILE")
        file_group.add_argument("-f", "--file", help="Filename of the config "
                                                     "file from which the experiment details"
                                                     " should be read.")
        # option 2: specify all experiment details as arguments
        detail_group = parser.add_argument_group("DETAILS")
        detail_group.add_argument("-p", "--platform", help="Specify "
                                                           "'local' or 'celery'")
        detail_group.add_argument('--data', help="Data in the following"
                                                 "format: trainfile,testfile,d,r such that "
                                                 "a data file can be found in "
                                                 "datadir/trainfile/Fold1/train.txt",
                                  type=str, nargs="+")
        detail_group.add_argument('--um', nargs="+")
        detail_group.add_argument('--uma', help="",
                                  type=str, nargs="+")
        detail_group.add_argument('--analysis', nargs="*")
        detail_group.add_argument('--data_dir')
        detail_group.add_argument('--output_base')
        detail_group.add_argument('--experiment_name')
        detail_group.add_argument("-r", "--rerun", action="store_true",
                                  help="Rerun last experiment.",
                                  default=False)
        detail_group.add_argument("--queue_name", type=str)

        args = parser.parse_known_args()[0]

        logging.basicConfig(format='%(asctime)s %(module)s: %(message)s',
                            level=logging.INFO)

        # determine whether to use config file or detailed args
        self.experiment_args = None
        if args.file:
            config_file = open(args.file)
            config = yaml.load(config_file, Loader=yaml.Loader)
            self.experiment_args = config
            random.seed(config.get('seed', 42))
            numpy.random.seed(config.get('seed', 42))
            config_file.close()
            try:
                self.meta_args = vars(parser.parse_known_args(
                    self.experiment_args["meta"].split())[0])
            except:
                parser.error("Please make sure there is a 'meta' section "
                             "present in the config file")
            # overwrite with command-line options if given
            for arg, value in vars(args).items():
                if value:
                    self.meta_args[arg] = value
        else:
            self.meta_args = vars(args)

        for k in list(self.meta_args.keys()) + ["meta"]:
            if k in self.experiment_args:
                del self.experiment_args[k]

        if self.meta_args["platform"] == "local":
            self.run = self.run_local
        elif self.meta_args["platform"] == "conf":
            self.run = self.run_conf
        else:
            parser.error("Please specify a valid platform.")

        usermodels = {}
        for umstr in self.meta_args["uma"]:
            parts = umstr.split(',')
            um, car = parts[:2]
            car = int(car)
            if len(parts) != car * 2 + 2:
                parser.error("Error in uma")
            p_click = ", ".join(parts[2:2 + car])
            p_stop = ", ".join(parts[2 + car:])
            if not um in usermodels:
                usermodels[um] = {}
            usermodels[um][car] = "--p_click %s --p_stop %s" % \
                                  (p_click, p_stop)

        basedir = os.path.join(os.path.abspath(self.meta_args["output_base"]),
                               self.meta_args["experiment_name"])

        i = 0
        while os.path.exists(os.path.join(basedir, "v%03d" % i)):
            i += 1
        if i > 0 and self.meta_args["rerun"]:
            i -= 1
        logging.info("Running experiment v%03d" % i)
        basedir = os.path.join(basedir, "v%03d" % i)
        if not os.path.exists(basedir):
            os.makedirs(basedir)
        logging.info("Results appear in %s" % basedir)

        config_bk = os.path.join(basedir, "meta_config_bk.yml")
        config_bk_file = open(config_bk, "w")
        yaml.dump(self.meta_args,
                  config_bk_file,
                  default_flow_style=False,
                  Dumper=yaml.Dumper)
        config_bk_file.close()

        skip = 0
        self.configurations = []
        for run_id in range(self.experiment_args["num_runs"]):
            for um in self.meta_args["um"]:
                for dstr in self.meta_args["data"]:
                    dparts = dstr.split(',')
                    data, d, r = dparts[:3]
                    d, r = int(d), int(r)
                    user_model_args = usermodels[um][r]
                    folds = glob.glob(os.path.join(
                        os.path.abspath(self.meta_args["data_dir"]),
                        data,
                        "Fold*"))
                    for fold in folds:
                        args = self.experiment_args.copy()
                        if len(dparts) > 3:
                            selected_weights = ",".join(dparts[3:])
                            args["system_args"] += " --selected_weights " + \
                                                   selected_weights
                        args["data_dir"] = self.meta_args["data_dir"]
                        args["fold_dir"] = fold
                        #            args["run_id"] = run_id
                        args["feature_count"] = d
                        args["user_model_args"] = user_model_args
                        args["output_dir"] = os.path.join(basedir,
                                                          'output',
                                                          um,
                                                          data,
                                                          os.path.basename(fold))
                        args["output_prefix"] = os.path.basename(fold)
                        args["run_id"] = run_id
                        if self.meta_args["rerun"]:
                            if not os.path.exists(os.path.join(
                                    args["output_dir"],
                                            "%s-%d.txt.gz" %
                                            (args["output_prefix"],
                                             run_id))):
                                self.configurations.append(args)
                            else:
                                skip += 1
                        else:
                            self.configurations.append(args)
        logging.info("Created %d configurations (and %d skipped)" % (
            len(self.configurations),
            skip))
        self.analytics = []
        for analyse in self.meta_args["analysis"]:
            aclass = get_class(analyse)
            a = aclass(basedir)
            self.analytics.append(a)

    def update_analytics(self):
        logging.info("Updating analytics for all existing log files.")
        for a in self.analytics:
            a.update()

    def update_analytics_file(self, log_file):
        for a in self.analytics:
            a.update_file(log_file)

    def finish_analytics(self):
        for a in self.analytics:
            a.finish()

    def store(self, conf, r):
        if not os.path.exists(conf["output_dir"]):
            try:
                os.makedirs(conf["output_dir"])
            except:
                pass
        log_file = os.path.join(conf["output_dir"], "%s-%d.txt.gz" %
                                (conf["output_prefix"], conf["run_id"]))

        log_fh = gzip.open(log_file, "wb")
        yaml.dump(r, log_fh, encoding='utf-8', default_flow_style=False, Dumper=yaml.Dumper)
        log_fh.close()
        return log_file

    def run_conf(self):
        if self.meta_args["rerun"]:
            self.update_analytics()

        logging.info("Creating log files %d tasks locally" % len(self.configurations))
        for conf in self.configurations:
            train = glob.glob(os.path.join(conf["fold_dir"], "*train.txt*"))[0]
            test = glob.glob(os.path.join(conf["fold_dir"], "*testset.txt*"))[0]
            conf["test_queries"] = test
            conf["training_queries"] = train

            if not os.path.exists(conf["output_dir"]):
                try:
                    os.makedirs(conf["output_dir"])
                except:
                    pass
            config_bk = os.path.join(conf["output_dir"], "config_bk.yml")
            config_bk_file = open(config_bk, "w")
            yaml.dump(conf,
                      config_bk_file,
                      default_flow_style=False,
                      Dumper=yaml.Dumper)
            config_bk_file.close()
        logging.info("Done")

    def run_local(self):
        if self.meta_args["rerun"]:
            self.update_analytics()

        logging.info("Running %d tasks locally" % len(self.configurations))
        for conf in self.configurations:
            train = glob.glob(os.path.join(conf["fold_dir"], "*trainingset.txt*"))[0]
            test = glob.glob(os.path.join(conf["fold_dir"], "*testset.txt*"))[0]
            conf["test_queries"] = test
            conf["training_queries"] = train

            if not os.path.exists(conf["output_dir"]):
                try:
                    os.makedirs(conf["output_dir"])
                except:
                    pass
            config_bk = os.path.join(conf["output_dir"], "config_bk.yml")
            config_bk_file = open(config_bk, "w")
            yaml.dump(conf,
                      config_bk_file,
                      default_flow_style=False,
                      Dumper=yaml.Dumper)
            config_bk_file.close()
            e = GenericExperiment("-f " + config_bk)
            r = e.run_experiment(None)
            log_file = self.store(conf, r)
            self.update_analytics_file(log_file)
            self.finish_analytics()
            logging.info("Done with %s, run %d" %
                         (conf["output_dir"], conf["run_id"]))
        logging.info("Done")
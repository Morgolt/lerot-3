try:
    import sys
    sys.path.append("..\\.")
except:
    pass
from lerot.experiment import MetaExperiment

if __name__ == "__main__":

    experiment = MetaExperiment()
    # todo: run summarization during meta experiment
    # todo: DCM
    experiment.run()



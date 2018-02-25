
try:
    from include import *
except:
    pass
from lerot.experiment import MetaExperiment

if __name__ == "__main__":
    experiment = MetaExperiment()
    # todo: fix condig creation  in code (debug)
    experiment.run()
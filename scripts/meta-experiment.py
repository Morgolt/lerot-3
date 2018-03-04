
try:
    from include import *
except:
    pass
from lerot.experiment import MetaExperiment

if __name__ == "__main__":
    experiment = MetaExperiment()
    # todo: fix config creation  in code (debug)
    experiment.run()
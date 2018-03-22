import numpy as np


class AbstractInterleavedComparison:

    def interleave(self, r1, r2, query, length):
        raise NotImplementedError("The derived class needs to implement "
                                  "interleave.")

    def interleave_n(self, r1, r2, query, length, num_repeat_interleaving):
        """ Default implementation just calls interleave n times. """
        return [self.interleave(r1, r2, query, length) for i in np.arange(num_repeat_interleaving)]

    def infer_outcome(self, l, a, c, query):
        raise NotImplementedError("The derived class needs to implement infer_outcome.")

import numpy as np
from numpy.linalg import norm


def rank(x, ties, reverse=False):
    n = len(x)
    if ties == "first":
        ix = zip(x, reversed(range(n)), range(n))
    elif ties == "last":
        ix = zip(x, range(n), range(n))
    elif ties == "random":
        ix = zip(x, np.random.permutation(n), range(n))
    else:
        raise Exception("Unknown method for breaking ties: \"%s\"" % ties)
    ix = list(ix)
    ix.sort(reverse=reverse)
    indexes = [i for _, _, i in ix]
    return [i for _, i in sorted(zip(indexes, range(n)))]


def sample_unit_sphere(n):
    """See http://mathoverflow.net/questions/24688/efficiently-sampling-
    points-uniformly-from-the-surface-of-an-n-sphere"""
    v = np.random.randn(n)
    v /= norm(v)
    return v
from importlib import import_module

import numpy as np
from numpy.linalg import norm


def rank(x: np.array, reverse=False):
    temp = np.argsort(x)
    ranks = np.empty_like(temp)
    ranks[temp] = np.arange(len(x))
    return ranks

def sample_unit_sphere(n):
    """See http://mathoverflow.net/questions/24688/efficiently-sampling-
    points-uniformly-from-the-surface-of-an-n-sphere"""
    v = np.random.randn(n)
    v /= norm(v)
    return v


def split_arg_str(arg_str):
    s = []
    max_index = 0
    while max_index < len(arg_str):
        index = arg_str.find("\"", max_index)
        # no more quotes: split the remaining args
        if index == -1:
            s.extend(arg_str[max_index:].split())
            break
        # quote found: find end + add preceding and quoted elements
        else:
            if index > max_index:
                s.extend(arg_str[max_index:index].split())
            closing_index = arg_str.find("\"", index + 1)
            if closing_index == -1:
                raise ValueError("Argument string contains non-matched quotes:"
                                 " %s" % arg_str)
            s.append(arg_str[index + 1:closing_index])
            max_index = closing_index + 1
    return s


def get_class(name):
    """Dynamically import lerot.<name>.

    Here be dragons.
    """
    module, classname = name.rsplit(".", 1)
    module = "lerot." + module

    try:
        return getattr(import_module(module), classname)
    except AttributeError as e:
        msg = ('%s while trying to import %r from %r'
               % (e.args[0], classname, module))
        e.args = (msg,) + e.args[1:]
        raise e


def get_cosine_similarity(v1, v2):
    """Compute the cosine similarity between two vectors."""
    if norm(v1) == 0 or norm(v2) == 0:
        return 0.0
    return np.dot(v1, v2) / (norm(v1) * norm(v2))

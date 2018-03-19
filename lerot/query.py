import pandas as pd
from sklearn.datasets import load_svmlight_file

from lerot.document import Document


class Query:
    __qid__ = None
    __feature_vectors__ = None
    __labels__ = None
    __predictions__ = None
    # document ids will be initialized as zero-based, os they can be used to
    # retrieve labels, predictions, and feature vectors
    __docids__ = None

    def __init__(self, qid, feature_vectors, labels=None):
        self.__qid__ = qid
        self.__feature_vectors__ = feature_vectors
        self.__labels__ = labels
        self.__docids__ = [Document(x) for x in range(len(labels))]

    def get_qid(self):
        return self.__qid__

    def get_docids(self):
        return self.__docids__

    def get_document_count(self):
        return len(self.__docids__)

    def get_feature_vectors(self):
        return self.__feature_vectors__

    def set_feature_vector(self, docid, feature_vector):
        self.__feature_vectors__[docid.get_id()] = feature_vector

    def get_feature_vector(self, docid):
        return self.__feature_vectors__[docid.get_id()]

    def get_labels(self):
        return self.__labels__

    def set_label(self, docid, label):
        self.__labels__[docid.get_id()] = label

    def get_label(self, docid):
        return self.__labels__[docid.get_id()]

    def set_labels(self, labels):
        self.__labels__ = labels

    def get_predictions(self):
        return self.__predictions__

    def get_prediction(self, docid):
        if self.__predictions__:
            return self.__predictions__[docid.get_id()]
        return None

    def set_predictions(self, predictions):
        self.__predictions__ = predictions


class QueryStream:
    __fn__ = None
    __num_features__ = 0

    def __init__(self, fn, num_features):
        self.__fn__ = fn
        self.__num_features__ = num_features

    def read_all(self):
        queries = {}
        x, y, qid = load_svmlight_file(self.__fn__, n_features=self.__num_features__, query_id=True)
        df = pd.DataFrame(data=x)
        df['y'] = y
        df['qid'] = qid
        for name, group in df.groupby(by='qid'):
            queries[name] = Query(name, group.drop(['qid', 'y'], axis=1).values, group.y.values)
        return queries


class Queries:
    """a list of queries with some convenience functions"""
    __num_features__ = 0
    __queries__ = None

    # cache immutable query values
    __qids__ = None
    __feature_vectors__ = None
    __labels__ = None

    def __init__(self, fh, num_features):
        self.__queries__ = QueryStream(fh, num_features).read_all()
        self.__num_features__ = num_features

    def __iter__(self):
        return iter(self.__queries__.values())

    def __getitem__(self, index):
        return self.get_query(index)

    def __len__(self):
        return len(self.__queries__)

    def keys(self):
        return self.__queries__.keys()

    def values(self):
        return self.__queries__.values()

    def get_query(self, index):
        return self.__queries__[index]

    def get_qids(self):
        if not self.__qids__:
            self.__qids__ = [query.get_qid() for query in self]
        return self.__qids__

    def get_labels(self):
        if not self.__labels__:
            self.__labels__ = [query.get_labels() for query in self]
        return self.__labels__

    def get_feature_vectors(self):
        if not self.__feature_vectors__:
            self.__feature_vectors__ = [query.get_feature_vectors() for query in self]
        return self.__feature_vectors__

    def set_predictions(self):
        raise NotImplementedError("Not yet implemented")

    def get_predictions(self):
        if not self.__predictions__:
            self.__predictions__ = [query.get_predictions() for query in self]
        return self.__predictions__

    def get_size(self):
        return self.__len__()


def load_queries(filename, features):
    """Utility method for loading queries from a file."""
    queries = Queries(filename, features)
    return queries

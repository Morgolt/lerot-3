class Document:

    def __init__(self, docid):
        self.docid = docid

    def __lt__(self, other):
        return self.docid < other.docid

    def __le__(self, other):
        return self.docid <= other.docid

    def __gt__(self, other):
        return self.docid > other.docid

    def __ge__(self, other):
        return self.docid >= other.docid

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return self.docid

    def __repr__(self):
        return 'Document@%d' % (self.docid)

    def __str__(self):
        return self.__repr__()

    def get_id(self):
        return self.docid

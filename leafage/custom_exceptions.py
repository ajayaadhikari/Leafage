class OneClassValues(Exception):
    def __init__(self):
        Exception.__init__(self, "The black box labels of the training-set does not have two classes.")

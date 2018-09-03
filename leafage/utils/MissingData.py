import numpy as np
from sklearn.preprocessing import Imputer


class MissingData:
    possible_strategies = ["drop_columns", "impute"]

    def __init__(self, strategy, columns_without_nan=None):
        if strategy not in self.possible_strategies:
            raise ValueError("Strategy %s is not supported. Choose from %s" % (strategy, self.possible_strategies))

        self.strategy = strategy
        self.columns_without_nan = columns_without_nan

    def fit(self, X, y=None):
        if self.columns_without_nan is None:
            # Get the column numbers that have at least one nan value
            self.columns_without_nan = np.argwhere(np.any(np.isnan(X), axis=0) == False).flatten()

        if self.strategy == self.possible_strategies[1]:
            self.imputer = Imputer()
            self.imputer.fit(X)
        return self

    def transform(self, X):
        if self.strategy == self.possible_strategies[0]:
            return X[:, self.columns_without_nan]
        elif self.strategy == self.possible_strategies[1]:
            return self.imputer.transform(X)


def tests():
    a = np.array([[1, 2, np.nan, 4], [1, np.nan, np.nan, 5]])

    # Test drop columns strategy
    md = MissingData("drop_columns")
    md.fit(a)
    assert np.array_equal(md.transform(a), np.array([[1, 4], [1, 5]])), "Error in drop column strategy"

    # Test imputer
    md = MissingData("impute")
    md.fit(a)
    assert np.array_equal(md.transform(a), np.array([[1, 2, 4], [1, 2, 5]])), "Error in impute strategy"


if __name__ == "__main__":
    tests()
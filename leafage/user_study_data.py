import pandas as pd
import numpy as np
from use_cases.data import Data

from utils.MissingData import MissingData
from utils.EstimatorSelectionHelper import EstimatorSelectionHelper

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC

import matplotlib.pyplot as plt
models1 = {
    'RandomForestClassifier': RandomForestClassifier(),
    'AdaBoostClassifier': AdaBoostClassifier(),
    'GradientBoostingClassifier': GradientBoostingClassifier(),
    'SVC': SVC()
}

c = [0.00001, 0.0001, 0.001, 0.01, 0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000, 5000, 100000, 50000, 1000000]
params1 = {
    'RandomForestClassifier': {'n_estimators': [16, 32]},
    'AdaBoostClassifier':  {'n_estimators': [16, 32]},
    'GradientBoostingClassifier': {'n_estimators': [16, 32], 'learning_rate': [0.8, 1.0]},
    'SVC': [
       {'kernel': ['linear'], 'C': c},
       {'kernel': ['rbf'], 'C': c, 'gamma': [0.00001, 0.0001, 0.001, 0.01, 0.1, 1]},
    ]
}






class HousingAdapted(Data):
    def __init__(self):
        # Read data from file
        df = pd.read_csv("../data/housing/train.csv", na_values="NA")
        all_columns = list(df)

        print(pd.isnull(df).sum().values)
        print(["%s: %s" % (c, v) for c, v in zip(all_columns, pd.isnull(df).sum().values)])

        # Drop columns with at least one null value
        df.dropna(how="any", inplace=True, axis=1)
        columns_without_null = list(df)
        columns_with_null = set(all_columns) - set(columns_without_null)

        #print("All columns: %s" % all_columns)
        #print("Columns without null: %s" % columns_without_null)
        #print("Columns with null: %s" % columns_with_null)

        # Add all columns as feature vector expect the sale price
        feature_vector = df.iloc[:, 1:-1].values

        # Make the sale price discrete as ["low", "medium", "high"]
        target_vector = self.split(df.iloc[:, -1].values)

        # Set the column names as the feature names
        feature_names = list(df)[1:-1]

        Data.__init__(self, feature_vector, target_vector, feature_names, name="Housing")

        #self.classifier = self.set_classifier()

    def set_classifier(self):
        classifier = SVC(C=10,kernel="rbf", gamma=0.001)
        classifier.fit(self.scaled_feature_vector, self.target_vector)
        return classifier

    def get_global_feature_importance(self):
        # Build a forest and compute the feature importances
        forest = ExtraTreesClassifier(n_estimators=250,
                                      random_state=0)
        X = self.scaled_feature_vector
        y = self.target_vector

        feature_names = self.get_one_hot_encoded_feature_names()

        forest.fit(X, y)
        importances = forest.feature_importances_
        std = np.std([tree.feature_importances_ for tree in forest.estimators_],
                     axis=0)
        indices = np.argsort(importances)[::-1]

        # Print the feature ranking
        print("Feature ranking:")

        for f in range(X.shape[1]):
            print("%s. feature %d (%f)" % (feature_names[f], indices[f], importances[indices[f]]))

        # Plot the feature importances of the forest
        plt.figure()
        plt.title("Feature importances")
        plt.bar(range(X.shape[1]), importances[indices],
                color="r", yerr=std[indices], align="center")
        plt.xticks(range(X.shape[1]), indices)
        plt.xlim([-1, X.shape[1]])
        plt.show()

    def grid_search(self):
        # Best results: SVM rbf kernel, gamma: 0.001, C: 10
        self.esh = EstimatorSelectionHelper(models1, params1)
        self.esh.fit(self.scaled_feature_vector, self.target_vector)
        self.results = self.esh.score_summary()
        self.results.to_csv("../output/housing/black_box_results.csv", index=False)

    @staticmethod
    def split(sale_price):
        result = []
        first_threshold = 150000
        second_threshold = 200000

        for price in sale_price:
            if price <= first_threshold:
                result.append("low")
            elif first_threshold < price <= second_threshold:
                result.append("medium")
            else:
                result.append("high")
        return np.array(result)


if __name__ == "__main__":
    a = HousingAdapted()
    #a.get_global_feature_importance()

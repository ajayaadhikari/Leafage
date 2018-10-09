from sklearn.svm import SVC
from use_cases.housing import HousingDataSet
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from leafage import Leafage
from scenario import Classifier


class UserStudy:
    train_size = 0.7
    random_state = 11

    def __init__(self):
        self.train, self.test = self.get_train_test()
        self.classifier = self.set_classifier()

    def set_classifier(self):
        svc = SVC(C=10, kernel="rbf", gamma=0.001, probability=True)
        svc.fit(self.train.scaled_feature_vector, self.train.target_vector)

        encoder = lambda x: self.train.pre_process_object.transform(x, scale=True)
        self.classifier = Classifier(svc, encoder)

        print(accuracy_score(self.test.target_vector, self.classifier.predict(self.test.feature_vector)))
        return self.classifier

    def get_train_test(self):
        housing_data = HousingDataSet()
        train, test, labels_train, labels_test = train_test_split(housing_data.feature_vector,
                                                                  housing_data.target_vector,
                                                                  train_size=self.train_size,
                                                                  random_state=self.random_state,
                                                                  stratify=housing_data.target_vector)
        training_data = housing_data.copy(train, labels_train)
        testing_data = housing_data.copy(test, labels_test)
        return training_data, testing_data

    def fidelity(self):


    def get_instances(self):
        test_predictions = self.classifier.predict(self.test.feature_vector)
        correct_instances = self.test.feature_vector[test_predictions == self.test.target_vector]

        leafage = Leafage(self.train, self.classifier, self.random_state, "closest_enemy")

        #i = 2

        for i in range(1,5):
            explanation = leafage.explain(correct_instances[i], 5)
            explanation.visualize_instance(path="../output/new_housing_instance2_%s.png" % i)
            explanation.visualize_leafage(path="../output/new_housing_explanation2_%s.png" % i)

    def get_feature_importance(self):
        import numpy as np
        import matplotlib.pyplot as plt

        from sklearn.datasets import make_classification
        from sklearn.ensemble import ExtraTreesClassifier

        X = self.train.scaled_feature_vector
        y = self.train.target_vector

        # Build a forest and compute the feature importances
        forest = ExtraTreesClassifier(n_estimators=250,
                                      random_state=0)

        forest.fit(X, y)
        importances = forest.feature_importances_
        std = np.std([tree.feature_importances_ for tree in forest.estimators_],
                     axis=0)
        indices = np.argsort(importances)[::-1]

        # Print the feature ranking
        print("Feature ranking:")

        features = self.train.get_one_hot_encoded_feature_names()
        for f in range(X.shape[1]):
            print("Feature %s: %s" % (features[indices[f]], importances[indices[f]]))

        # Plot the feature importances of the forest
        plt.figure()
        plt.title("Feature importances")
        plt.bar(range(X.shape[1]), importances[indices],
                color="r", yerr=std[indices], align="center")
        plt.xticks(range(X.shape[1]), indices)
        plt.xlim([-1, X.shape[1]])
        plt.show()



if __name__ == "__main__":
    a = UserStudy()
    a.get_instances()

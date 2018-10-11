from sklearn.svm import SVC

from faithfulness import Faithfulness
from explanation import Explanation
from use_cases.housing import HousingDataSet
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from leafage import LeafageBinary
from scenario import Classifier
from local_model import Neighbourhood, Distances
import pandas as pd
import numpy as np
import time


class UserStudy:
    train_size = 0.7
    random_state = 11

    def __init__(self):
        self.train, self.test = self.get_train_test()
        self.classifier = self.set_classifier()

        predicted_training_data = self.classifier.predict(self.train.feature_vector)
        self.leafage = LeafageBinary(self.train, predicted_training_data, self.random_state, "closest_enemy")

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
        test_set_predictions = self.classifier.predict(self.test.feature_vector)
        faithfulness = Faithfulness(self.test, test_set_predictions, self.leafage.get_local_model, verbose=True)
        print(faithfulness)

    def get_instances(self):
        test_predictions = self.classifier.predict(self.test.feature_vector)
        correct_instances = self.test.feature_vector[test_predictions == self.test.target_vector]

        explanation_types = ["feature_based"]*10 + ["example_based"]*10 + ["leafage"]*10 + ["no_ex"]*10
        np.random.seed(0)

        # for i in range(5):
        #     test_instance = correct_instances[i]
        #     prediction = self.classifier.predict([test_instance])[0]
        #     explanation = self.leafage.explain(correct_instances[i], prediction, amount_of_examples=5)
        #     explanation.visualize_feature_importance(path="../output/user_study/test_test_%s.png" % i,
        #                                              show_values=False)

        num = 36
        for i in range(76, 80):
            e_type = "lol"
            num += 1
            test_instance = correct_instances[i]
            prediction = self.classifier.predict([test_instance])[0]
            explanation = self.leafage.explain(correct_instances[i], prediction, amount_of_examples=5)
            explanation.set_plotly_imports()

            explanation.visualize_instance(path="../output/user_study/%s_instance_prediction_%s.png" % (num, explanation.fact_class))
            if e_type == "feature_based":
                explanation.visualize_feature_importance(path="../output/user_study/%s_feature_importance.png" % num, show_values=False)
            elif e_type == "example_based":
                explanation.visualize_examples(path="../output/user_study/%s_examples.png" % num, type="both")
            elif e_type == "leafage":
                explanation.visualize_leafage(path="../output/user_study/%s_leafage.png" % num)
            else:
                print("Nope %s" % e_type)

            instance_to_test, instance_to_test_class = self.get_instance_to_test(explanation)
            figure = Explanation.visualize_instance_one_line(instance_to_test)
            Explanation.visualize_png(figure, path="../output/user_study/%s_test_instance_prediction_%s.png" % (num, instance_to_test_class))

    def get_instance_to_test(self, explanation):
        np.random.seed(int(time.time()))
        new_instance_class = np.random.choice(["High", "Low"], 1)[0]
        class_encoded = self.train.target_vector_encoder.transform([new_instance_class])[0]

        training_set = self.train.scaled_feature_vector
        predictions = self.classifier.predict(self.train.feature_vector)
        instance = explanation.local_model.instance_to_explain

        indices, _ = Neighbourhood.get_closest_instances_of_label(training_set,
                                                                          predictions,
                                                                          Distances.unbiased_distance_function,
                                                                          10,
                                                                          instance,
                                                                          class_encoded)
        closest_instances = self.train.feature_vector[indices].tolist()
        if explanation.fact_class == new_instance_class:
            cross_check_instances = explanation.original_in_support.tolist()
        else:
            cross_check_instances = explanation.original_against.tolist()

        found = False
        i = 0
        while not found:
            if closest_instances[i] not in cross_check_instances:
                result_instance = closest_instances[i]
                found = True
            i += 1

        result_instance = pd.Series(result_instance, index=self.train.feature_names)
        get_both_measure = lambda feet_square: "%s m<sup>2</sup> (%s ft<sup>2</sup>)" % (int(feet_square * 0.09290304), feet_square)
        result_instance["Living Area"] = get_both_measure(result_instance["Living Area"])
        return result_instance, new_instance_class


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

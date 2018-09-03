from leafage import LeafageBinary
from faithfulness import Faithfulness
from wrapper_lime import WrapperLime
from scenario import Scenario
from sklearn.model_selection import train_test_split
from itertools import combinations
from utils.Classifiers import train
from use_cases.all_use_cases import all_data_sets
import numpy as np
import pandas as pd
from custom_exceptions import OneClassValues
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_classification


setup_blackbox_models = [("lr", {}), ("svc", {"kernel": "linear", "probability": True}), ("lda", {}),
                         ("rf", {}), ("dt", {}),
                         ("nb_g", {}),
                         ("mlp", {})]


class EvaluateFaithfulness:
    random_state = 11

    def __init__(self, dataset, train_size):
        self.data = dataset
        self.train_size = train_size
        self.one_vs_rest = self.get_one_vs_rest()

    def get_one_vs_one(self):
        classes = np.unique(self.data.target_vector)
        indices_per_class = dict(zip(classes, [[] for x in range(len(classes))]))
        for index in range(len(self.data.target_vector)):
            indices_per_class[self.data.target_vector[index]].append(index)

        one_vs_one_data = {}
        combinations_classes = combinations(classes, 2)
        for class_1, class_2 in combinations_classes:
            merged_indices = indices_per_class[class_1] + indices_per_class[class_2]
            feature_vector = self.data.feature_vector[merged_indices]
            target_vector = self.data.target_vector[merged_indices]
            train, test, labels_train, labels_test = train_test_split(feature_vector,
                                                                      target_vector,
                                                                      train_size=self.train_size,
                                                                      random_state=self.random_state,
                                                                      stratify=target_vector)
            training_data = self.data.copy(train, labels_train)
            testing_data = self.data.copy(test, labels_test)
            one_vs_one_data[(class_1, class_2)] = (training_data, testing_data)

        return one_vs_one_data

    def get_one_vs_rest(self):
        classes = np.unique(self.data.target_vector)
        one_vs_all = {}
        for class_name in classes:
            binary_labels = np.array([class_name if x == class_name else -1 for x in self.data.target_vector])
            train, test, labels_train, labels_test = train_test_split(self.data.feature_vector,
                                                                      binary_labels,
                                                                      train_size=self.train_size,
                                                                      random_state=self.random_state,
                                                                      stratify=binary_labels)
            training_data = self.data.copy(train, labels_train)
            testing_data = self.data.copy(test, labels_test)
            one_vs_all[class_name] = (training_data, testing_data)

        return one_vs_all

    def get_faithfulness(self):
        dfs = []
        for classifier_name, variables in setup_blackbox_models:
            print("\tClassifier: %s" % classifier_name)
            try:
                df = self.get_faithfulness_classifier(classifier_name, variables)
                dfs.append(df)
            except OneClassValues:
                print("Classifier %s on dataset %s: only predicts one class" % (classifier_name, self.data.name))

        merged_df = pd.concat(dfs, ignore_index=True)
        path = "../output/result_faithfulness/dataset_%s_%s.csv" % (self.data.name, self.train_size)
        merged_df.to_csv(path, index=False)
        return merged_df

    def get_faithfulness_classifier(self, classifier_name, classifier_variables):
        radii = np.arange(0.05, 1.05, 0.05)

        def combine(container):
            f1_score = np.around(np.mean([x.average_f1_score_per_radius for x in container], axis=0), 2)
            std = np.around(np.mean([x.std_f1_score_per_radius for x in container], axis=0), 2)
            base_line = np.around(np.mean([x.average_base_line_per_radius for x in container], axis=0), 2)
            amount = np.around(np.mean([x.average_amount_per_radius for x in container], axis=0), 2)
            return [f1_score.tolist(), std.tolist(), base_line.tolist(), amount.tolist()]

        def create_df(l_ce_all, l_cb_all, lime_all):
            l_ce_all, l_cb_all, lime_all = combine(l_ce_all), combine(l_cb_all), combine(lime_all)
            len_radii = len(radii)+1
            amount_of_rows = len_radii*3

            dataset_name_column = [self.data.name]*amount_of_rows
            classifier_name_column = [classifier_name]*amount_of_rows
            #classifier_variables_column = [str(classifier_variables)]*amount_of_rows
            strategy_column = ["Leafage: closest enemy"]*len_radii + ["Leafage: closest boundary"]*len_radii + ["lime"]*len_radii
            accuracy_column = l_ce_all[0] + l_cb_all[0] + lime_all[0]
            std_column = l_ce_all[1] + l_cb_all[1] + lime_all[1]
            base_line_column = l_ce_all[2] + l_cb_all[2] + lime_all[2]
            radii_column = (radii.tolist()+[-1]) *3
            amount_column = l_ce_all[3] + l_cb_all[3] + lime_all[3]

            table = np.array([dataset_name_column, classifier_name_column, strategy_column,
                     accuracy_column, std_column, base_line_column, radii_column, amount_column]).transpose()
            column_names = ["Dataset name", "Classifier name", "Strategy",
                            "f1_score", "std", "class_of_z", "radius", "amount in radius"]
            return pd.DataFrame(data=table, columns=column_names)

        leafage_ce_all = []
        leafage_cb_all = []
        lime_all = []
        i = 0
        for name in self.one_vs_rest.keys():
            i += 1
            print("\t\tOne vs one %s/%s %s:" % (i, len(self.one_vs_rest.keys()), name))
            training_data, test_data = self.one_vs_rest[name]
            classifier = train(classifier_name,
                               training_data.one_hot_encoded_feature_vector,
                               training_data.target_vector,
                               classifier_variables)
            predicted_training_labels = classifier.predict(training_data.one_hot_encoded_feature_vector)

            test_accuracy = accuracy_score(test_data.target_vector, classifier.predict(test_data.one_hot_encoded_feature_vector))
            print("\t\tBlack-box test accuracy: %s" % test_accuracy)

            leafage_ce = LeafageBinary(training_data, predicted_training_labels, self.random_state, "closest_enemy")
            leafage_cb = LeafageBinary(training_data, predicted_training_labels, self.random_state, "closest_boundary")
            lime = WrapperLime(training_data, classifier.predict_proba)

            predicted_test_labels = classifier.predict(test_data.one_hot_encoded_feature_vector)

            lime_all.append(Faithfulness(test_data, predicted_test_labels, lime.get_local_model, radii))
            leafage_ce_all.append(Faithfulness(test_data, predicted_test_labels, leafage_ce.get_local_model, radii))
            leafage_cb_all.append(Faithfulness(test_data, predicted_test_labels, leafage_cb.get_local_model, radii))

        return create_df(leafage_ce_all, leafage_cb_all, lime_all)


def create_artificial_datasets():
    X, y = make_classification(n_samples=1000, n_features=2, n_redundant=0, n_informative=2)


def faithfulness_data_sets():
    train_size = 0.7
    all_df = []
    dataset_names = ["iris", "wine", "breast_cancer", "digits"]
    for name in dataset_names:
        dataset = all_data_sets[name]()
        print("Dataset %s" % name)
        all_df.append(EvaluateFaithfulness(dataset, train_size).get_faithfulness())

    all_df = pd.concat(all_df, ignore_index=True)
    path = "../output/result_faithfulness/result"
    all_df.to_csv("%s.csv" % path, index=False)


def housing_from_use_cases():
    scenario = Scenario("load_from_use_cases", "housing", "lr")
    leafage = scenario.leafage
    explanation = scenario.get_explanation(leafage.training_data.feature_vector[1], 5)
    explanation.visualize_feature_importance(amount_of_features=10, target="write_to_file", path="../output/feature_importance2.png")
    explanation.visualize_examples(target="write_to_file", path="../output/examples_in_support2.png", type="examples_in_support")
    explanation.visualize_examples(target="write_to_file", path="../output/examples_against2.png", type="examples_against")


def housing_from_file():
    scenario = Scenario("load_from_file", "../data/housing/pre_processed_train.csv", "lr")
    leafage = scenario.leafage
    explanation = scenario.get_explanation(scenario.data.feature_vector[0], amount_of_examples=5)
    a = 4


if __name__ == "__main__":
    housing_from_use_cases()

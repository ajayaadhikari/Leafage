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
from sklearn.metrics import f1_score
from sklearn.datasets import make_classification
from sklearn.svm import SVC
from use_cases.data import Data

setup_blackbox_models = [("lr", {}), ("svc", {"kernel": "linear", "probability": True}), ("lda", {}),
                         ("rf", {"n_estimators":10}), ("rf", {"n_estimators":50}),
                         ("rf", {"n_estimators":100}), ("rf", {"n_estimators":150}),
                         ("rf", {"n_estimators":200}),
                         ("dt", {"min_samples_split": 2}), ("dt", {"min_samples_split": 5}),
                         ("dt", {"min_samples_split": 10}), ("dt", {"min_samples_split": 15}),
                         ("dt", {"min_samples_split": 20}), ("dt", {"min_samples_split": 30}),
                         ("nb_g", {})]


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
            if len(classes) == 2:
                binary_labels = self.data.target_vector
                dataset_name = "%s vs %s" % tuple(self.data.target_vector_encoder.inverse_transform(classes))
            else:
                binary_labels = np.array([0 if x == class_name else 1 for x in self.data.target_vector])
                dataset_name = "%s vs rest" % self.data.target_vector_encoder.inverse_transform([class_name])[0]

            train, test, labels_train, labels_test = train_test_split(self.data.feature_vector,
                                                                      binary_labels,
                                                                      train_size=self.train_size,
                                                                      random_state=self.random_state,
                                                                      stratify=binary_labels)
            training_data = self.data.copy(train, labels_train)
            testing_data = self.data.copy(test, labels_test)
            one_vs_all[dataset_name] = (training_data, testing_data)

            if len(classes) == 2:
                break

        return one_vs_all

    def get_faithfulness(self, i):
        dfs = []
        for classifier_name, variables in setup_blackbox_models:
            print("\tClassifier: %s" % classifier_name)
            try:
                df = self.get_faithfulness_classifier(classifier_name, variables, i)
                dfs.append(df)
            except OneClassValues:
                print("Classifier %s on dataset %s: only predicts one class" % (classifier_name, self.data.name))

        merged_df = pd.concat(dfs, ignore_index=True)
        path = "../output/result_faithfulness/new_optimization_dataset_%s_%s_i_%s.csv" % (self.data.name, self.train_size, i)
        merged_df.to_csv(path, index=False)
        return merged_df

    def get_faithfulness_classifier(self, classifier_name, classifier_variables, i):
        def get_separability(feature_vector, target_vector):
            C = [0.00001, 0.0001, 0.001, 0.01, 0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000, 5000, 100000, 50000, 1000000]
            max_f1_score = 0

            for c_value in C:
                svc = SVC(kernel="linear", C=c_value)
                svc.fit(feature_vector, target_vector)
                predicted_labels = svc.predict(feature_vector)
                f1 = f1_score(target_vector, predicted_labels, average="macro")
                if f1 > max_f1_score:
                    max_f1_score = f1

            return max_f1_score

        def create_df(dataset_sub_name, test_data,
                      faithfulness_leafage_ce, faithfulness_leafage_cb,
                      faithfulness_lime, classifier_f1_score, separability):
            amount_per_strategy = len(test_data.feature_vector)
            amount_of_rows = amount_per_strategy * 4

            i_column = [i]*amount_of_rows
            order_column = range(1, amount_per_strategy+1) * 4
            dataset_name_column = [self.data.name] * amount_of_rows
            dataset_sub_name_column = [dataset_sub_name] * amount_of_rows
            classifier_name_column = [classifier_name] * amount_of_rows
            classifier_variables_column = [str(classifier_variables)] * amount_of_rows
            classifier_f1_score_column = [str(classifier_f1_score)] * amount_of_rows
            separability_column = [str(separability)] * amount_of_rows
            strategy_column = ["Leafage: Closest Enemy"]*amount_per_strategy + \
                              ["Leafage: Closest Boundary"]*amount_per_strategy + \
                              ["Lime"]*amount_per_strategy + \
                              ["Baseline"]*amount_per_strategy
            f1_score_column = faithfulness_leafage_ce.f1 + faithfulness_leafage_cb.f1 + \
                              faithfulness_lime.f1 + faithfulness_lime.base_line_f1
            auc_column = faithfulness_leafage_ce.auc + faithfulness_leafage_cb.auc + \
                              faithfulness_lime.auc + faithfulness_lime.base_line_auc
            amount_column = faithfulness_leafage_ce.amount * 4
            total_amount_column = [str(amount_per_strategy)] * amount_of_rows

            table = np.array([i_column, order_column, dataset_name_column, dataset_sub_name_column, classifier_name_column,
                              classifier_variables_column, classifier_f1_score_column, separability_column,
                              strategy_column, amount_column, total_amount_column, f1_score_column, auc_column]).transpose()
            column_names = ["i","Order", "Dataset Name", "Class vs Class", "Classifier Name",
                            "Classifier Variables", "Classifier F1", "Separability", "Strategy",
                            "Amount within", "Total test amount", "F1 Score method", "AUC score"]

            return pd.DataFrame(data=table, columns=column_names)

        resulting_dfs = []
        j = 0
        for name in self.one_vs_rest.keys():
            j += 1
            print("\t\tOne vs one %s/%s %s:" % (j, len(self.one_vs_rest.keys()), name))

            training_data, test_data = self.one_vs_rest[name]
            train_feature_vector = training_data.one_hot_encoded_feature_vector
            test_feature_vector = test_data.one_hot_encoded_feature_vector

            classifier = train(classifier_name,
                               train_feature_vector,
                               training_data.target_vector,
                               classifier_variables)
            predicted_training_labels = classifier.predict(train_feature_vector)
            predicted_scores = np.array([x[1] for x in classifier.predict_proba(train_feature_vector)])
            predicted_test_labels = classifier.predict(test_feature_vector)

            if len(np.unique(predicted_training_labels)) <= 1 or len(np.unique(predicted_test_labels)) <= 1:
                raise OneClassValues()

            classifier_f1_score = f1_score(test_data.target_vector,
                                           predicted_test_labels, average="macro")
            separability = get_separability(test_feature_vector,
                                            predicted_test_labels)
            print("\t\tBlack-box test f1_score: %s" % classifier_f1_score)
            print("\t\tSeparability: %s" % separability)

            leafage_ce = LeafageBinary(training_data, predicted_training_labels, self.random_state, "closest_enemy", i=i)
            leafage_cb = LeafageBinary(training_data, predicted_training_labels, self.random_state, "closest_boundary", i=i)
            lime = WrapperLime(training_data, classifier.predict_proba)

            faithfulness_leafage_ce = Faithfulness(test_data, predicted_test_labels, leafage_ce.get_local_model)
            faithfulness_leafage_cb = Faithfulness(test_data, predicted_test_labels, leafage_cb.get_local_model)
            faithfulness_lime = Faithfulness(test_data, predicted_test_labels, lime.get_local_model)

            resulting_dfs.append(create_df(name, test_data,
                                           faithfulness_leafage_ce, faithfulness_leafage_cb,
                                           faithfulness_lime, classifier_f1_score, separability))

        return pd.concat(resulting_dfs, ignore_index=True)


def create_artificial_datasets():
    datasets = []
    random_state = 11
    amount_of_features = 2
    for i in np.arange(0.1,1.1,0.1):
        X, y = make_classification(n_samples=600, n_features=2, n_redundant=0,
                                   n_informative=2, class_sep=i, random_state=random_state)
        dataset = Data(X, y, range(amount_of_features), name="Separability: %s" % i)
        datasets.append(dataset)
    return datasets


def faithfulness_data_sets():
    train_size = 0.7
    all_df = []
    real_datasets = [all_data_sets[name]() for name in ["iris", "wine", "breast_cancer", "bank_note", "abalone"]]
    artifical_datasets = create_artificial_datasets()
    datasets = artifical_datasets + real_datasets
    for i in range(1, 20):
        current_i = []
        for dataset in datasets:
            print("Dataset %s" % dataset.name)
            faithfulness = EvaluateFaithfulness(dataset, train_size).get_faithfulness(i)
            current_i.append(faithfulness)

        current_i = pd.concat(current_i, ignore_index=True)
        path = "../output/result_faithfulness/faithfulness_i_%s.csv" % i
        current_i.to_csv(path, index=False)

        all_df.append(current_i)

    all_df = pd.concat(all_df, ignore_index=True)
    path = "../output/result_faithfulness/faithfulness_all.csv"
    all_df.to_csv(path, index=False)


def housing_from_use_cases():
    scenario = Scenario("load_from_use_cases", "housing", "lr")
    leafage = scenario.leafage
    explanation = scenario.get_explanation(leafage.training_data.feature_vector[1], 5)
    explanation.visualize_feature_importance(amount_of_features=10, target="write_to_file", path="../output/feature_importance2.png")
    explanation.visualize_examples(target="write_to_file", path="../output/examples_in_support2.png", type="examples_in_support")
    explanation.visualize_examples(target="write_to_file", path="../output/examples_against2.png", type="examples_against")


def breast_cancer_from_use_cases():
    scenario = Scenario("load_from_use_cases", "breast_cancer", "lr")
    leafage = scenario.leafage
    explanation = scenario.get_explanation(leafage.training_data.feature_vector[1], 5)
    explanation.visualize_feature_importance(amount_of_features=10, target="write_to_file", path="../output/breast_cancer_feature_importance.png")
    explanation.visualize_examples(amount_of_features=10,target="write_to_file", path="../output/breast_cancer_examples_in_support.png", type="examples_in_support")
    explanation.visualize_examples(amount_of_features=10,target="write_to_file", path="../output/breast_cancer_examples_against.png", type="examples_against")

def housing_from_file():
    scenario = Scenario("load_from_file", "../data/housing/pre_processed_train.csv", "lr")
    leafage = scenario.leafage
    explanation = scenario.get_explanation(scenario.data.feature_vector[0], amount_of_examples=5)
    a = 4


if __name__ == "__main__":
    faithfulness_data_sets()

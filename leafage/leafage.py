import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from explanation import Explanation
from use_cases.all_use_cases import all_data_sets
from wrapper_lime import WrapperLime
from faithfulness import Faithfulness
from local_model import LocalModel
from utils.MathFunctions import euclidean_distance
from utils import Classifiers

import matplotlib.pyplot as plt


class Leafage:
    def __init__(self,
                 training_data,
                 testing_data,
                 predict,
                 predict_proba,
                 random_state,
                 neighbourhood_sampling_strategy):
        """
        Full form of LEAFAgE: Local Example And Feature-based model Agnostic Explanation.

        :param training_data: Should be of type use_cases.data.Data
        :param testing_data: Should be of type use_cases.data.Data
        :param predict: Predict function of the black-box classifier
        :param predict_proba: Predict probability per class of the black-box classifier
        :param random_state
        :param neighbourhood_sampling_strategy: Either "closest boundary" or "closest instance"
        """

        self.training_data = training_data
        self.testing_data = testing_data
        self.predict = predict
        self.predict_proba = predict_proba
        self.random_state = random_state
        self.neighbourhood_sampling_strategy = neighbourhood_sampling_strategy

        self.explanatory_examples = LeafageMultiClass(self.training_data,
                                                      self.predict(self.training_data.feature_vector),
                                                      self.random_state,
                                                      self.neighbourhood_sampling_strategy)
        if self.testing_data is not None:
            self.leafage_faithfulness, self.lime_faithfulness = \
                self.explanatory_examples.get_faithfulness(self.testing_data,
                                                           self.predict(self.testing_data.feature_vector),
                                                           np.arange(0.1, 1.1, 0.1),
                                                           self.lime_local_model)

    def explain(self, test_instance, amount_of_examples):
        # Get the instances to explain
        return self.explanatory_examples.explain(test_instance, self.predict_proba([test_instance])[0], amount_of_examples)

    def lime_local_model(self, training_data, classes):
        lime = WrapperLime(training_data, self.predict_proba, classes)
        return lambda instance, _: lime.get_local_model(instance)


class LeafageMultiClass:
    def __init__(self, training_data, predicted_labels, random_state,
                 neighbourhood_sampling_strategy):
        self.training_data = training_data
        self.predicted_labels = predicted_labels
        self.random_state = random_state
        self.labels = np.unique(predicted_labels)
        self.neighbourhood_sampling_strategy = neighbourhood_sampling_strategy

        self.one_vs_rest = {}
        if len(self.labels) == 1:
            print("Data with only one class %s" % self.labels[0])
        elif self.is_binary():
            leafage_binary = LeafageBinaryClass(self.training_data, predicted_labels,
                                             random_state, self.neighbourhood_sampling_strategy)
            self.one_vs_rest[self.labels[0]] = leafage_binary
            self.one_vs_rest[self.labels[1]] = leafage_binary
        else:
            self.one_vs_rest = self.get_one_vs_all(self.training_data, self.predicted_labels)

    def get_one_vs_all(self, training_data, predicted_labels):
        one_vs_all = {}
        for i, label in enumerate(predicted_labels):
            binary_predicted_labels = self.labels_one_vs_all(label, predicted_labels)
            one_vs_all[label] = LeafageBinaryClass(training_data, binary_predicted_labels,
                                                   self.random_state, self.neighbourhood_sampling_strategy)
        return one_vs_all

    @staticmethod
    def labels_one_vs_all(target_label, predicted_labels):
        """
        Convert the non target_label in predicted labels to -1
        """
        return np.array([target_label if x == target_label else -1 for x in predicted_labels])

    def get_one_vs_one(self, fact_class, foil_class):
        indices_fact_foil = np.where(np.logical_or(self.predicted_labels == fact_class, self.predicted_labels == foil_class))
        new_feature_vector = self.training_data.feature_vector[indices_fact_foil]
        new_real_labels = self.training_data.target_vector[indices_fact_foil]
        new_predicted_labels = self.predicted_labels[indices_fact_foil]
        return LeafageBinaryClass(self.training_data.copy(new_feature_vector, new_real_labels),
                                  new_predicted_labels, self.random_state, self.neighbourhood_sampling_strategy)

    def is_binary(self):
        return len(self.labels) == 2

    def get_faithfulness(self, test_set, test_set_predictions, radii, lime_get_local_model):
        leafage_faithfulness = {}
        lime_faithfulness = {}

        for label in self.labels:
            print("Label %s vs rest" % label)
            one_vs_rest_predictions = self.labels_one_vs_all(label, test_set_predictions)
            print("Evaluate Leafage!")
            leafage_faithfulness[label] = Faithfulness(test_set,
                                                    one_vs_rest_predictions,
                                                    self.one_vs_rest[label].get_local_model,
                                                    radii)
            print("Evaluate LIME")
            classes = sorted(np.unique(one_vs_rest_predictions))
            lime_get_local_model = lime_get_local_model(self.training_data, classes)
            lime_faithfulness[label] = Faithfulness(test_set,
                                                    one_vs_rest_predictions,
                                                    lime_get_local_model,
                                                    radii)
        return leafage_faithfulness, lime_faithfulness

    def explain(self, test_instance, probabilities_per_class, amount_of_examples):
        sorted_indices = np.argsort(probabilities_per_class)
        fact_class = sorted_indices[-1]
        foil_class = sorted_indices[-2]
        leafage_binary = self.get_one_vs_one(fact_class, foil_class)
        return leafage_binary.explain(test_instance, fact_class, amount_of_examples)


class LeafageBinaryClass:
    # Static variables
    distance_measure = euclidean_distance
    linear_classifier = LogisticRegression
    linear_classifier_variables = {}

    def __init__(self, training_data, predicted_labels, random_state, neighbourhood_sampling_strategy):
        self.training_data = training_data

        self.predicted_labels = predicted_labels
        self.random_state = random_state
        self.neighbourhood_sampling_strategy = neighbourhood_sampling_strategy

    def explain(self, test_instance, test_instance_prediction, amount_of_examples=10):
        pre_process = lambda X: self.training_data.pre_process(X, scale=True)
        #pre_process = lambda X: self.training_data.pre_process(X, scale=False)

        scaled_training_set = self.training_data.scaled_feature_vector
        scaled_test_instance = pre_process([test_instance])[0]
        #scaled_training_set = self.training_data.feature_vector
        #scaled_test_instance = test_instance

        np.random.seed(self.random_state)
        local_model = LocalModel(scaled_test_instance,
                                 test_instance_prediction,
                                 scaled_training_set,
                                 self.predicted_labels,
                                 pre_process,
                                 self.neighbourhood_sampling_strategy)

        # Get the closest instances
        indices_examples_in_support, _ = \
            local_model.neighbourhood.get_examples_in_support(amount_of_examples,
                                                              local_model.distances.get_final_distance,
                                                              self.training_data.scaled_feature_vector,
                                                              self.training_data.target_vector)
        indices_examples_against, _ = \
            local_model.neighbourhood.get_examples_against(amount_of_examples,
                                                           local_model.distances.get_final_distance,
                                                           self.training_data.scaled_feature_vector,
                                                           self.training_data.target_vector)
        examples_in_support = self.training_data.feature_vector[indices_examples_in_support]
        examples_against = self.training_data.feature_vector[indices_examples_against]

        inverse_transform_label = lambda x: "rest" if x == -1 else self.training_data.target_vector_encoder.inverse_transform(x)
        enemy_class = local_model.neighbourhood.enemy_class
        fact_class = inverse_transform_label(test_instance_prediction)
        foil_class = inverse_transform_label(enemy_class)
        coefficients = self.normalize(self.filter_coefficients(local_model.linear_model.coefficients, scaled_test_instance))

        a = Explanation(test_instance,
                        examples_in_support,
                        examples_against,
                        coefficients,
                        fact_class,
                        foil_class,
                        self.training_data.feature_names,
                        local_model)
        return a

    def filter_coefficients(self, one_hot_encoded_coefficients, one_hot_encoded_instance):
        """
        Remove from coefficients the indices that do not correspond to the categorical value of the given instance
        """
        has_categorical_features = self.training_data.pre_process_object.has_categorical_features
        if has_categorical_features:
            one_hot_encoder = self.training_data.pre_process_object.one_hot_encoder
            end_index = one_hot_encoder.feature_indices_[-1]
            instance_categorical_values_indices = np.where(one_hot_encoded_instance[0:end_index] == 1)
            coefficients_categorical_features = one_hot_encoded_coefficients[instance_categorical_values_indices]

            result = one_hot_encoded_coefficients[end_index:].tolist()
            categorical_features = self.training_data.pre_process_object.categorical_features
            for new_value, index in zip(coefficients_categorical_features, categorical_features):
                result.insert(index, new_value)
            return result
        else:
            return one_hot_encoded_coefficients

    @staticmethod
    def normalize(container):
        total = float(sum(container))
        return np.array([x/total for x in container])


    def get_local_model(self, instance, prediction):
        return self.explain(instance, prediction, 1).local_model.linear_model

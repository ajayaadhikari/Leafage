import numpy as np
from sklearn.linear_model import LogisticRegression

from src.leafage.explanatory_examples import SetupExplanatoryExamples, Explanation
from src.leafage.wrapper_lime import WrapperLime
from src.leafage.faithfulness import Faithfulness
from src.leafage.local_model import LocalModel
from src.utils.MathFunctions import euclidean_distance

import matplotlib.pyplot as plt


class MeemBinaryClass:
    # Static variables
    distance_measure = euclidean_distance
    linear_classifier = LogisticRegression
    linear_classifier_variables = {}

    def __init__(self, training_data, predicted_labels, random_state, neighbourhood_sampling_strategy):
        self.training_data = training_data

        self.predicted_labels = predicted_labels
        self.random_state = random_state
        self.neighbourhood_sampling_strategy = neighbourhood_sampling_strategy

    def get_explanatory_examples(self, test_instance, number_examples=5):
        pass

    def explain(self, test_instance, test_instance_prediction, amount_of_examples=10):
        pre_process = lambda X: self.training_data.pre_process(X, scale=True)
        inverse_pre_process = lambda X: self.training_data.inverse_pre_process(X, scale=True)

        scaled_training_set = self.training_data.scaled_feature_vector
        scaled_test_instance = pre_process([test_instance])[0]

        np.random.seed(self.random_state)
        local_model = LocalModel(scaled_test_instance,
                                 test_instance_prediction,
                                 scaled_training_set,
                                 self.predicted_labels,
                                 pre_process,
                                 self.neighbourhood_sampling_strategy)
        print("\t%s" % local_model.faithfulness.accuracy)

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

        a = Explanation(test_instance,
                        examples_in_support,
                        examples_against,
                        local_model)
        return a

    def get_local_model(self, instance, prediction):
        return self.explain(instance, prediction, 1).local_model.linear_model

    def visualize_faithfulness(self):
        for i in np.arange(1):
            pass

    def visualize_table(self, instance):
        pass

    def visualize(self, test_instance):
        explanation = self.explain(test_instance)
        self.visualize_2(explanation)

    def visualize_2(self, explanation, amount_of_features=10):
        positive_color = "yellowgreen"
        negative_color = "tomato"

        amount_of_features = min(amount_of_features, len(explanation.original_test_instance))
        columns = self.training_data.feature_names
        labels = np.append(explanation.predicted_labels_example_instances,[1])
        labels_colors = [negative_color if label == 0 else positive_color for label in labels]

        data = np.vstack((explanation.original_test_instance, explanation.original_example_instances))[:, :amount_of_features]

        # Sort the features according to the absolute value in descending order
        sort_index = np.flip(np.argsort(np.abs(explanation.linear_model.linear_model_coefficients)), axis=0)[:amount_of_features]
        sorted_coefficients = np.around(np.array(explanation.linear_model.linear_model_coefficients)[sort_index], decimals=2)
        sorted_feature_names = np.array(self.training_data.feature_names)[sort_index]

        # Set the negative coefficients as red and the positive as green
        colors = [negative_color if coef < 0 else positive_color for coef in sorted_coefficients]
        sorted_coefficients = np.abs(sorted_coefficients)

        index = np.arange(amount_of_features) + 0.3
        bar_width = 0.4

        plt.bar(index, sorted_coefficients, bar_width, color=colors)

        # Add a table at the bottom of the axes
        the_table = plt.table(cellText=data,
                              rowLabels=labels,
                              colLabels=sorted_feature_names,
                              rowColours=labels_colors,
                              loc='bottom')

        # Adjust layout to make room for the table:
        plt.subplots_adjust(left=0.2, bottom=0.3)

        plt.ylabel("Influence on the target variable")
        plt.title('Influence on the prediction')
        plt.xticks([])
        plt.ylim([0, 1])

        plt.show()


class MeemMultiClass:
    def __init__(self, training_data, predicted_labels, random_state,
                 lime_get_local_model, neighbourhood_sampling_strategy):
        self.training_data = training_data
        self.predicted_labels = predicted_labels
        self.random_state = random_state
        self.labels = np.unique(predicted_labels)
        self.lime_get_local_model = lime_get_local_model
        self.neighbourhood_sampling_strategy = neighbourhood_sampling_strategy

        self.one_vs_rest = {}
        if len(self.labels) == 1:
            print("Data with only one class %s" % self.labels[0])
        elif self.is_binary():
            meem_binary = MeemBinaryClass(self.training_data, predicted_labels,
                                          random_state, self.neighbourhood_sampling_strategy)
            self.one_vs_rest[self.labels[0]] = meem_binary
            self.one_vs_rest[self.labels[1]] = meem_binary
        else:
            self.one_vs_rest = self.get_one_vs_all(self.training_data, self.predicted_labels)

    def get_one_vs_all(self, training_data, predicted_labels):
        one_vs_all = {}
        for i, label in enumerate(predicted_labels):
            binary_predicted_labels = self.labels_one_vs_all(label, predicted_labels)
            one_vs_all[label] = MeemBinaryClass(training_data, binary_predicted_labels,
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
        return MeemBinaryClass(self.training_data.copy(new_feature_vector, new_real_labels),
                               new_predicted_labels, self.random_state, self.neighbourhood_sampling_strategy)

    def is_binary(self):
        return len(self.labels) == 2

    def get_faithfulness(self, test_set, test_set_predictions, radii):
        meem_faithfulness = {}
        lime_faithfulness = {}

        for label in self.labels:
            print("Label %s vs rest" % label)
            one_vs_rest_predictions = self.labels_one_vs_all(label, test_set_predictions)
            print("Evaluate MEEM!")
            meem_faithfulness[label] = Faithfulness(test_set,
                                                    one_vs_rest_predictions,
                                                    self.one_vs_rest[label].get_local_model,
                                                    radii)
            print("Evaluate LIME")
            classes = sorted(np.unique(one_vs_rest_predictions))
            lime_get_local_model = self.lime_get_local_model(self.training_data, classes)
            lime_faithfulness[label] = Faithfulness(test_set,
                                                    one_vs_rest_predictions,
                                                    lime_get_local_model,
                                                    radii)
        return meem_faithfulness, lime_faithfulness

    def explain(self, test_instance, probabilities_per_class):
        sorted_indices = np.argsort(probabilities_per_class)
        fact_class = sorted_indices[-1]
        foil_class = sorted_indices[-2]
        binary_meem = self.get_one_vs_one(fact_class, foil_class)
        return binary_meem.explain(test_instance, fact_class)


class SetupExplanatoryExamplesMeem(SetupExplanatoryExamples):
    def __init__(self, setup_variables):
        SetupExplanatoryExamples.__init__(self, setup_variables)
        self.explanatory_examples = MeemMultiClass(self.training_data,
                                                   self.predict(self.training_data.feature_vector),
                                                   self.setup_variables.random_state,
                                                   self.lime_local_model,
                                                   self.setup_variables.neighbourhood_sampling_strategy)
        if self.setup_variables.train_size < 1:
            self.meem_faithfulness, self.lime_faithfulness = self.explanatory_examples.get_faithfulness(self.testing_data,
                                                                                                    self.predict(self.testing_data.feature_vector),
                                                                                                    np.arange(0.1, 1.1, 0.1))

    def explain(self, test_instance):
        # Get the instances to explain
        return self.explanatory_examples.explain(test_instance, self.predict_proba([test_instance])[0])

    def lime_local_model(self, training_data, classes):
        lime = WrapperLime(training_data, self.predict_proba, classes)
        return lambda instance, _: lime.get_local_model(instance)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import src.utils.Classifiers
from src.use_cases.all_use_cases import different_data_sets
from src.use_cases.data import Data
import numpy as np


class Explanation:
    def __init__(self,
                 original_test_instance,
                 original_example_instances,
                 predicted_labels_example_instances):
        self.original_test_instance = original_test_instance
        self.original_example_instances = original_example_instances
        self.predicted_labels_example_instances = predicted_labels_example_instances


class SetupVariables:
    def __init__(self,
                 dataset_name,
                 train_size,
                 random_state,
                 classifier_name,
                 classifier_var={},
                 ee_method_var={},
                 neighbourhood_sampling_strategy="closest_boundary"):
        self.dataset_name = dataset_name
        self.train_size = train_size
        self.random_state = random_state
        self.classifier_name = classifier_name
        self.classifier_var = classifier_var
        self.ee_method_var = ee_method_var
        self.neighbourhood_sampling_strategy = neighbourhood_sampling_strategy
        self.other = {}
        print("Strategy:%s" % neighbourhood_sampling_strategy)

    def add_setup_variable(self, variable_name, value):
        self.other[variable_name] = value

    @staticmethod
    def str_dict(dictionary):
        """
        Convert {"a":1,"b":2} to a_1_b_2
        """
        if dictionary is None or len(dictionary) == 0:
            return ""
        else:
            return reduce(lambda i, j: "%s_%s" % (i, j), ["%s_%s" % (x, dictionary[x]) for x in dictionary.keys()])

    def __str__(self):
        """
        Get a unique name of this setup
        """
        return "%s_%s_%s_%s_%s_%s" % (self.dataset_name, self.train_size, self.random_state, self.classifier_name, self.str_dict(self.classifier_var), self.str_dict(self.other))


class SetupExplanatoryExamples:
    def __init__(self,
                 setup_variables):
        self.setup_variables = setup_variables
        self.data = different_data_sets[setup_variables.dataset_name]()
        self.predict, self.predict_proba,  self.training_data, self.test, self.labels_test = self.setup()

    def setup(self):
        # Split in train en test
        train, test, labels_train, labels_test = train_test_split(self.data.feature_vector,
                                                                  self.data.target_vector,
                                                                  train_size=self.setup_variables.train_size,
                                                                  random_state=self.setup_variables.random_state,
                                                                  stratify=self.data.target_vector)
        encoder = self.data.encoder
        np.random.seed(self.setup_variables.random_state)
        if encoder is None:
            # Train classifier
            classifier = src.utils.Classifiers.train(self.setup_variables.classifier_name, train,
                                                     labels_train, self.setup_variables.classifier_var)
            predict_proba = classifier.predict_proba
            predict = classifier.predict
        else:
            encoded_train = encoder.transform(train)
            classifier = src.utils.Classifiers.train(self.setup_variables.classifier_name, encoded_train,
                                                     labels_train, self.setup_variables.classifier_var)
            predict_proba = lambda x: classifier.predict_proba(encoder.transform(x))
            predict = lambda x: classifier.predict(encoder.transform(x))

        print(accuracy_score(labels_test, predict(test)))

        # Compute explanations on the training set
        training_data = Data(train, labels_train, self.data.feature_names,
                             self.data.class_names, self.data.categorical_features,
                             self.data.categorical_names, self.data.preprocessing_method)
        return predict, predict_proba, training_data, test, labels_test

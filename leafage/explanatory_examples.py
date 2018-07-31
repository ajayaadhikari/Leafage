from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import utils.Classifiers
from use_cases.all_use_cases import different_data_sets
import numpy as np


class Explanation:
    def __init__(self,
                 test_instance,
                 examples_in_support,
                 examples_against,
                 local_model):

        self.test_instance = test_instance,
        self.examples_in_support = examples_in_support,
        self.examples_against = examples_against
        self.local_model = local_model


class SetupVariables:
    def __init__(self,
                 dataset_name,
                 train_size,
                 random_state,
                 classifier_name,
                 classifier_var={},
                 neighbourhood_sampling_strategy="closest_boundary"):
        self.dataset_name = dataset_name
        self.train_size = train_size
        self.random_state = random_state
        self.classifier_name = classifier_name
        self.classifier_var = classifier_var
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
        self.test_faithfulness = self.setup_variables.train_size < 1
        self.data = different_data_sets[setup_variables.dataset_name]()
        self.predict, self.predict_proba, self.training_data, self.testing_data = self.setup()

    def setup(self):
        # Split in train en test
        if self.test_faithfulness:
            train, test, labels_train, labels_test = train_test_split(self.data.feature_vector,
                                                                      self.data.target_vector,
                                                                      train_size=self.setup_variables.train_size,
                                                                      random_state=self.setup_variables.random_state,
                                                                      stratify=self.data.target_vector)
        else:
            train, labels_train = self.data.feature_vector, self.data.target_vector
            test, labels_test = [],[]
        input_encoder_black_box = self.data.input_encoder_black_box
        np.random.seed(self.setup_variables.random_state)

        # Train the classifier
        encoded_train = input_encoder_black_box(train)
        classifier = utils.Classifiers.train(self.setup_variables.classifier_name, encoded_train,
                                                         labels_train, self.setup_variables.classifier_var)
        predict_proba = lambda X: classifier.predict_proba(input_encoder_black_box(X))
        predict = lambda X: classifier.predict(input_encoder_black_box(X))

        if self.test_faithfulness:
            print(accuracy_score(labels_test, predict(test)))
            testing_data = self.data.copy(test, labels_test)
        else:
            testing_data = None

        # Make a new data object of the training-set
        training_data = self.data.copy(train, labels_train)
        return predict, predict_proba, training_data, testing_data

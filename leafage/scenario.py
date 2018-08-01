from sklearn.metrics import accuracy_score
import numpy as np

from leafage import Leafage
from use_cases.read_from_file import FileDataSet
from utils import Classifiers
from utils.StringFunctions import str_dict_snake
from sklearn.model_selection import train_test_split
from use_cases.all_use_cases import all_data_sets


class Scenario:
    def __init__(self,
                 dataset_source,
                 dataset_name,
                 train_size,
                 random_state,
                 classifier_name,
                 classifier_hyper_parameters={},
                 neighbourhood_sampling_strategy="closest_boundary",
                 encoder_classifier=None):
        """
        :param dataset_source: Either "read_from_file" or "load_from_use_cases".
        :param dataset_name: if dataset_source="read_from_file" then dataset_name is the path to the file
                             if dataset_source="load_from_use_cases" then dataset_name should be in use_cases.all_use_cases.different_data_sets
        :param train_size: Denote in range of 0 to 1 the ratio of the training-set size.
        :param random_state
        :param classifier_name: Short name of the classifier, should be in $leafage.utils.Classifiers
        :param classifier_hyper_parameters: As a dictionary e.g. {"C": 5, "kernel": "linear"}
        :param neighbourhood_sampling_strategy: Either "closest boundary" or "closest instance"
        :param encoder_classifier: The encoder function that converts an instance such that it can be fed to the black-box classifier
                                   If None then the categorical features are one-hot-encoded before feeding to the classifier
                                   and numerical features will be left untouched
        """
        self.dataset_source = dataset_source
        self.dataset_name = dataset_name
        self.train_size = train_size
        self.random_state = random_state
        self.classifier_name = classifier_name
        self.classifier_hyper_parameters = classifier_hyper_parameters
        self.neighbourhood_sampling_strategy = neighbourhood_sampling_strategy
        print("Strategy:%s" % neighbourhood_sampling_strategy)

        self.test_faithfulness = self.train_size < 1
        self.data = self.get_data()

        self.encoder_classifier = encoder_classifier
        if encoder_classifier is None:
            self.encoder_classifier = self.data.pre_process_object.transform

        self.predict, self.predict_proba, self.training_data, self.testing_data = self.setup()
        self.leafage = Leafage(self.training_data, self.testing_data, self.predict, self.predict_proba,
                               random_state, self.neighbourhood_sampling_strategy)

    def get_leafage_object(self):
        return self.leafage

    def get_data(self):
        if self.dataset_source == "load_from_use_cases":
            return all_data_sets[self.dataset_name]()
        else:
            return FileDataSet(self.dataset_name)

    def setup(self):
        # Split in train en test
        if self.test_faithfulness:
            train, test, labels_train, labels_test = train_test_split(self.data.feature_vector,
                                                                      self.data.target_vector,
                                                                      train_size=self.train_size,
                                                                      random_state=self.random_state,
                                                                      stratify=self.data.target_vector)
        else:
            train, labels_train = self.data.feature_vector, self.data.target_vector
            test, labels_test = None, None

        # Train the classifier
        np.random.seed(self.random_state)
        encoded_train = self.encoder_classifier(train)
        classifier = Classifiers.train(self.classifier_name, encoded_train,
                                       labels_train, self.classifier_hyper_parameters)

        # Get the predict and the predict_proba functions
        predict_proba = lambda X: classifier.predict_proba(self.encoder_classifier(X))
        predict = lambda X: classifier.predict(self.encoder_classifier(X))

        if self.test_faithfulness:
            print("Accuracy: %s" % accuracy_score(labels_test, predict(test)))
            testing_data = self.data.copy(test, labels_test)
        else:
            testing_data = None

        # Make a new data object of the training-set
        training_data = self.data.copy(train, labels_train)
        return predict, predict_proba, training_data, testing_data

    def __str__(self):
        """
        Get a unique name of this setup
        """
        return "%s_%s_%s_%s_%s" % (self.dataset_name, self.train_size, self.random_state, self.classifier_name,
                                      str_dict_snake(self.classifier_hyper_parameters))



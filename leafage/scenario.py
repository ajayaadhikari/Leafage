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
                 dataset,
                 classifier_name,
                 classifier_hyper_parameters={},
                 random_state=11,
                 neighbourhood_sampling_strategy="closest_enemy",
                 encoder_classifier=None):
        """
        This class is a wrapper around the class leafage.Leafage.
        It contains code to prepare the data such that it can be fed to leafage.Leafage.
        It is possible to load data in three ways namely from file, from the module use_cases and directly by providing
        an instance of the class utils.data.Data
        :param dataset_source: Either "load_from_file" or "load_from_use_cases" or "data_object"
        :param dataset: if dataset_source="load_from_file" then dataset is the path to the file, the last column should
                                           be the target vector and the rest the feature vector
                        if dataset_source="load_from_use_cases" then dataset should be in use_cases.all_use_cases.different_data_sets
                        if dataset_source="data_object" then dataset should be an object of utils.data.Data
        :param classifier_name: Short name of the classifier, should be in $leafage.utils.Classifiers
        :param classifier_hyper_parameters: As a dictionary e.g. {"C": 5, "kernel": "linear"}
        :param random_state
        :param neighbourhood_sampling_strategy: Either "closest boundary" or "closest instance"
        :param encoder_classifier: The encoder function that converts an instance such that it can be fed to the black-box classifier
                                   If None then the categorical features are one-hot-encoded before feeding to the classifier
                                   and numerical features will be left untouched
        """
        self.dataset_source = dataset_source
        self.random_state = random_state
        self.classifier_name = classifier_name
        self.classifier_hyper_parameters = classifier_hyper_parameters
        self.neighbourhood_sampling_strategy = neighbourhood_sampling_strategy
        print("Strategy: Leafage %s" % neighbourhood_sampling_strategy)

        self.data = self.__get_data(dataset)

        self.encoder_classifier = encoder_classifier
        if encoder_classifier is None:
            self.encoder_classifier = self.data.pre_process_object.transform

        self.classifier = Classifier(self.__get_classifier(), self.encoder_classifier)
        self.leafage = Leafage(self.data, self.classifier,
                               random_state, self.neighbourhood_sampling_strategy)

    def get_leafage_object(self):
        return self.leafage

    def get_explanation(self, instance, amount_of_examples, type="contrastive"):
        return self.leafage.explain(instance, amount_of_examples, type)

    def __get_data(self, dataset):
        if self.dataset_source == "load_from_use_cases":
            return all_data_sets[dataset]()
        elif self.dataset_source == "data_object":
            return dataset
        elif self.dataset_source == "load_from_file":
            return FileDataSet(dataset)
        else:
            raise ValueError("Scenario does not support mode %s" % self.dataset_source)

    def __get_classifier(self):
        # Train the classifier
        np.random.seed(self.random_state)
        encoded_train = self.encoder_classifier(self.data.feature_vector)
        return Classifiers.train(self.classifier_name, encoded_train,
                                       self.data.target_vector, self.classifier_hyper_parameters)

    def __str__(self):
        """
        Get a unique name of this setup
        """
        return "%s_%s_%s" % (self.random_state, self.classifier_name,
                            str_dict_snake(self.classifier_hyper_parameters))


class Classifier:
    def __init__(self, classifier, encoder=None):
        self.classifier = classifier
        self.encoder = encoder

    def predict(self, X):
        if self.encoder is None:
            return self.classifier.predict(X)
        else:
            return self.classifier.predict(self.encoder(X))

    def predict_proba(self, X):
        if self.encoder is None:
            return self.classifier.predict_proba(X)
        else:
            return self.classifier.predict_proba(self.encoder(X))

import warnings
warnings.filterwarnings(action='ignore', category=DeprecationWarning)

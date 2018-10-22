from collections import Counter

from sklearn.metrics import f1_score

from custom_exceptions import OneClassValues
from utils.Evaluate import EvaluationMetrics
from utils.MathFunctions import euclidean_distance
from sklearn.svm import SVC
from math import exp
import numpy as np


class LocalModel:
    linear_classifier_type = SVC
    linear_classifier_variables = {"kernel": "linear"}

    def __init__(self, instance_to_explain, prediction, training_set,
                 black_box_labels, pre_process, neighbourhood_strategy,
                 data_preprocessed=True, i=None):

        if len(np.unique(black_box_labels)) <= 1:
            raise OneClassValues()
        if not data_preprocessed:
            training_set = pre_process(training_set)
            instance_to_explain = pre_process([instance_to_explain])[0]

        self.instance_to_explain = instance_to_explain
        self.prediction = prediction
        self.training_set = training_set
        self.black_box_labels = black_box_labels
        self.pre_process = pre_process
        self.neighbourhood_strategy = neighbourhood_strategy

        self.neighbourhood = Neighbourhood(self.instance_to_explain, self.prediction,
                                           self.training_set, self.black_box_labels, strategy=neighbourhood_strategy, i=i)
        self.valid = self.neighbourhood.valid
        if self.valid:
            self.distances, self.linear_model = self.build_model()
            self.faithfulness = self.get_faithfulness()

    def build_model(self):
        # Build local regression model
        local_classifier = self.linear_classifier_type(**self.linear_classifier_variables)
        local_classifier.fit(self.neighbourhood.instances, self.neighbourhood.labels, self.neighbourhood.weights)

        # Move line such that it goes through the test instance
        moved_intercept = -1 * np.dot(self.instance_to_explain, np.transpose(local_classifier.coef_[0]))
        classes = local_classifier.classes_
        coefficients = local_classifier.coef_[0]

        linear_model = LinearModel(coefficients,
                                   local_classifier.intercept_,
                                   self.pre_process,
                                   moved_intercept=moved_intercept,
                                   classes=classes)
        distance_to_enemy = self.neighbourhood.get_distance_to_closest_opposite_instance\
                            (Distances.unbiased_distance_function, self.training_set)

        return Distances(self.instance_to_explain, linear_model, distance_to_enemy), linear_model

    def get_classifier(self):
        feature_vector, target_vector = self.neighbourhood.instances, self.neighbourhood.labels
        C = [0.00001, 0.0001, 0.001, 0.01, 0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000, 5000, 100000, 50000, 1000000]
        max_f1_score = 0

        for c_value in C:
            svc = SVC(kernel="linear", C=c_value)
            svc.fit(feature_vector, target_vector)
            predicted_labels = svc.predict(feature_vector)
            f1 = f1_score(target_vector, predicted_labels, average="macro")
            if f1 > max_f1_score:
                max_f1_score = f1
                classifier = svc
        return classifier

    def get_faithfulness(self):
        local_model_predictions = self.linear_model.get_predictions(self.neighbourhood.instances, pre_process=False)
        return EvaluationMetrics(self.neighbourhood.labels, local_model_predictions, sample_weight=self.neighbourhood.weights)

    def get_diameter(self):
        return max([self.distances.get_unbiased_distance(x) for x in self.neighbourhood.instances])

    def get_predictions(self, instances):
        return self.linear_model.get_predictions(instances)


class LinearModel:
    def __init__(self, coefficients, original_intercept, pre_process, threshold=0, moved_intercept=None, classes=[0, 1]):
        self.coefficients = coefficients
        self.original_intercept = original_intercept
        self.pre_process = pre_process
        self.threshold = threshold
        self.moved_intercept = moved_intercept
        self.classes = classes

    def get_predictions(self, instances, pre_process=True):
        """
            Get the predictions made by the local_model on the instances in the given neighbourhood
        """
        if pre_process:
            instances = self.pre_process(instances)

        local_model_predictions = []
        for instance in instances:
            regression_value = np.dot(np.array(self.coefficients), np.array(instance)) + self.original_intercept
            prediction = self.classes[1 if regression_value > self.threshold else 0]
            local_model_predictions.append(prediction)

        return local_model_predictions

    def get_scores(self, instances, pre_process=True):
        """
            Get the predictions made by the local_model on the instances in the given neighbourhood
        """
        if pre_process:
            instances = self.pre_process(instances)

        local_model_scores = []
        for instance in instances:
            regression_value = np.dot(np.array(self.coefficients), np.array(instance)) + self.original_intercept
            local_model_scores.append(regression_value)

        return local_model_scores


class Neighbourhood:
    min_weight = 0.01

    def __init__(self, instance_to_explain, prediction, training_set, black_box_labels, strategy="closest_boundary", i=3):
        if i is None:
            i = 3

        self.instance_to_explain = instance_to_explain
        self.prediction = prediction

        classes = np.array(sorted(np.unique(black_box_labels)))
        self.enemy_class = classes[classes != self.prediction][0]

        if strategy == "closest_enemy":
            get_neighbourhood = self.get_neighbourhood_around_closest_enemy
        elif strategy == "closest_boundary":
            get_neighbourhood = self.get_neighbourhood_of_closest_boundary
        else:
            raise ValueError("%s not supported" % strategy)

        self.instances, self.labels, self.weights = get_neighbourhood(training_set, black_box_labels, i=i)
        self.valid = self.is_valid()

    def get_neighbourhood_oud(self, training_feature_vector, black_box_labels):

        # Get the weight of each training instance
        weights = [self.get_weight(instance) for instance in training_feature_vector]

        indexed_weights = [(index, weights[index]) for index in range(len(weights))]
        #filtered_weights = filter(lambda x: x[1] > Neighbourhood.min_weight, indexed_weights)

        # Get the black-box labels of the corresponding instances of the filtered_weights
        indices = [x[0] for x in indexed_weights]
        neighbourhood = training_feature_vector[indices]
        neighbourhood_labels = black_box_labels[indices]
        neighbourhood_weights = np.array(weights)[indices]

        return neighbourhood, neighbourhood_labels, neighbourhood_weights

    def get_neighbourhood_around_closest_enemy(self, training_set, black_box_labels, i=3):
        unbiased_distance = Distances.unbiased_distance_function

        closest_enemy = Neighbourhood.get_closest_enemy_instance(training_set, black_box_labels, unbiased_distance, self.instance_to_explain, self.prediction)

        # Get the unbiased distance from each training instance to the closest enemy
        distances = np.array(map(lambda x: unbiased_distance(x, closest_enemy), training_set))

        # Amount per party: number of dimension
        amount_per_class = i*len(training_set[0])

        # Get the indices of each label
        indices_label_0 = np.where(black_box_labels == self.prediction)
        indices_label_1 = np.where(black_box_labels == self.enemy_class)

        # Split the training set in classes
        distances_0 = distances[indices_label_0]
        instances_0 = training_set[indices_label_0]
        distances_1 = distances[indices_label_1]
        instances_1 = training_set[indices_label_1]

        # Get the closest instances from both classes
        indices_sorted_label_0 = np.argsort(distances_0)[:amount_per_class]
        indices_sorted_label_1 = np.argsort(distances_1)[:amount_per_class]

        neighbourhood = np.concatenate((instances_0[indices_sorted_label_0], instances_1[indices_sorted_label_1]))
        neighbourhood_labels = np.concatenate((np.full(len(indices_sorted_label_0), self.prediction),
                                               np.full(len(indices_sorted_label_1), self.enemy_class)))
        weights = np.ones(len(neighbourhood_labels))

        return neighbourhood, neighbourhood_labels, weights

    def get_neighbourhood_of_closest_boundary(self, training_set, black_box_labels, i=3):
        d = len(training_set[0])
        amount_per_class_big_neighbourhood = 3*i*d
        amount_per_class_small_neighbourhood = i*d
        unbiased_distance = Distances.unbiased_distance_function

        closest_enemy = Neighbourhood.get_closest_enemy_instance(training_set, black_box_labels, unbiased_distance, self.instance_to_explain, self.prediction)

        # Get the neighbourhood around the closet enemy
        big_neighbourhood, big_neighbourhood_labels, _ = self.get_neighbourhood_around_closest_enemy(training_set, black_box_labels, amount_per_class_big_neighbourhood)

        # Split the big neighbourhood in classes
        instances_0 = big_neighbourhood[np.where(big_neighbourhood_labels == self.prediction)]
        instances_1 = big_neighbourhood[np.where(big_neighbourhood_labels == self.enemy_class)]

        # Get closest instance from the opposite class of each training instance
        closest_instances_0 = map(lambda instance: Neighbourhood.__get_closest_instances(instances_1, unbiased_distance, 1, instance)[1][0], instances_0)
        closest_instances_1 = map(lambda instance: Neighbourhood.__get_closest_instances(instances_0, unbiased_distance, 1, instance)[1][0], instances_1)

        # Get the distance to the closest instance from the opposite class of each training instance
        distances_to_closest_instance_0 = map(lambda i: unbiased_distance(instances_0[i], closest_instances_0[i]), range(len(instances_0)))
        distances_to_closest_instance_1 = map(lambda i: unbiased_distance(instances_1[i], closest_instances_1[i]), range(len(instances_1)))

        # Get the distance to the the closest enemy
        distances_to_closest_enemy_0 = map(lambda i: unbiased_distance(instances_0[i], closest_enemy),range(len(instances_0)))
        distances_to_closest_enemy_1 = map(lambda i: unbiased_distance(instances_1[i], closest_enemy),range(len(instances_1)))

        # Combine distance to the both distances
        final_distance_0 = map(lambda i: distances_to_closest_instance_0[i] + distances_to_closest_enemy_0[i], range(len(instances_0)))
        final_distance_1 = map(lambda i: distances_to_closest_instance_1[i] + distances_to_closest_enemy_1[i], range(len(instances_1)))
        #final_distance_0 = distances_to_closest_instance_0
        #final_distance_1 = distances_to_closest_instance_1

        # Get the top instances per class with the shortest distance
        indices_sorted_label_0 = np.argsort(final_distance_0)[:amount_per_class_small_neighbourhood]
        indices_sorted_label_1 = np.argsort(final_distance_1)[:amount_per_class_small_neighbourhood]

        #
        neighbourhood = np.concatenate((instances_0[indices_sorted_label_0], instances_1[indices_sorted_label_1]))
        neighbourhood_labels = np.concatenate((np.full(len(indices_sorted_label_0), self.prediction),
                                               np.full(len(indices_sorted_label_1), self.enemy_class)))
        weights = np.ones(len(neighbourhood_labels))

        np.concatenate((np.full(len(indices_sorted_label_0), self.prediction),
                        np.full(len(indices_sorted_label_1), self.enemy_class)))

        return neighbourhood, neighbourhood_labels, weights

    def get_weight(self, training_instance):
        return Distances.get_weight(training_instance, self.instance_to_explain, self.sigma)

    @staticmethod
    def __get_closest_instances(target_instances, distance_function, amount, source):
        # From the filtered instances get the closest instances
        distances = map(lambda x: distance_function(x, source), target_instances)
        sort_index = np.argsort(distances)[:amount]

        return sort_index, np.array(target_instances[sort_index])

    @staticmethod
    def __get_closest_instances_of_label(target_instances, target_instances_labels, distance_function, amount, source, label):
        # Get the instances with the given label from the target_instances
        label_index = np.where(target_instances_labels == label)[0]
        filtered_instances = target_instances[label_index]

        closest_instances_index, closest_instances = Neighbourhood.__get_closest_instances(filtered_instances, distance_function, amount, source)

        return label_index[closest_instances_index], closest_instances

    @staticmethod
    def get_closest_enemy_instance(target_instances, target_instances_labels, distance_function, source, label):
        return Neighbourhood.__get_closest_instances_of_label(target_instances, target_instances_labels, distance_function, 1, source, Neighbourhood.__get_enemy_class(label, target_instances_labels))[1]

    @staticmethod
    def __get_distance_to_closest_enemy_instance(target_instances, target_instances_labels, distance_function, source, label):
        instance = Neighbourhood.get_closest_enemy_instance(target_instances, target_instances_labels, distance_function, source, label)
        return distance_function(instance, source)

    def get_closest_instances(self, amount, distance_function, label, target_instances=None, labels=None):
        if target_instances is None or labels is None:
            target_instances = self.instances
            labels = self.labels
        # Get the instances with the given label from the current neighbourhood
        return self.__get_closest_instances_of_label(target_instances, labels, distance_function, amount, self.instance_to_explain, label)

    def get_examples_in_support(self, amount, distance_function, target_instances=None, labels=None):
        return self.get_closest_instances(amount, distance_function, self.prediction, target_instances, labels)

    def get_examples_against(self, amount, distance_function, target_instances=None, labels=None):
        return self.get_closest_instances(amount, distance_function, self.enemy_class, target_instances, labels)

    def get_distance_to_closest_opposite_instance(self, distance_function, target_instances=None, labels=None):
        instance = self.get_examples_against(1, distance_function, target_instances, labels)[1]
        return distance_function(instance, self.instance_to_explain)

    def is_valid(self):
        #min_amount = len(self.instance_to_explain)
        min_amount = 1
        counts = Counter(self.labels)
        if self.enemy_class in counts and counts[self.enemy_class] >= min_amount:
            return True
        else:
            return False

    @staticmethod
    def __get_enemy_class(label, all_labels):
        classes = np.array(sorted(np.unique(all_labels)))
        return classes[classes != label][0]


class Distances:
    def __init__(self, test_instance, linear_model, distance_to_enemy):
        self.test_instance = test_instance
        self.coefficients = linear_model.coefficients
        self.intercept = linear_model.moved_intercept
        self.distance_to_enemy = distance_to_enemy

    # Dummy for compatibility if two instances are given, second one will be ignored
    def get_black_box_distance(self, training_instance, dummy=None):
        """
            This distance measure gives biased importance to feature dimensions according to the
            linear model
        """
        nominator = abs(np.dot(training_instance, np.transpose(self.coefficients)) + self.intercept)
        denominator = np.sqrt(np.dot(self.coefficients, np.transpose(self.coefficients)))
        return nominator/float(denominator)

    def get_biased_distance(self, training_instance, dummy=None):
        difference = np.power(training_instance - np.array(self.test_instance), 2)
        dot_product = np.dot(np.abs(self.coefficients), difference.transpose())
        square = np.sqrt(dot_product)
        return square

    def get_unbiased_distance(self, training_instance, dummy=None):
        """
            This distance measure gives equal importance to each feature dimension
        """
        return self.unbiased_distance_function(training_instance, self.test_instance)

    def get_final_distance(self, training_instance, dummy=None):
        """
            This distance measure combines the unbiased distance and the black-box distance
        """
        d = len(training_instance)
        unbiased_distance = self.get_unbiased_distance(training_instance)
        black_box_distance = np.sqrt(d* self.get_black_box_distance(training_instance))

        return unbiased_distance + black_box_distance

    @staticmethod
    def get_weight(training_instance, test_instance, sigma):
        """
            Get the weight (range: 0-1) based on the unbiased distance between training_instance and self.test_instance
            The growth of the weight given the distance is determined by $self.sigma
        """
        distance = Distances.unbiased_distance_function(training_instance, test_instance)
        return exp(- (distance ** 2) / (sigma ** 2))

    @staticmethod
    def unbiased_distance_function(x, y):
        return euclidean_distance(x, y)

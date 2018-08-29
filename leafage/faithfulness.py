from collections import Counter

import numpy as np
import matplotlib.pyplot as plt

from local_model import Distances, Neighbourhood
from utils.Evaluate import EvaluationMetrics


class Faithfulness:
    a = 0.95

    def __init__(self, test_set, test_predictions, function_get_local_model, radii, verbose=False):
        self.scale = lambda x: test_set.pre_process([x], scale=True)[0]
        self.test_set = test_set.feature_vector
        self.scaled_test_set = test_set.scaled_feature_vector
        self.test_predictions = test_predictions
        self.function_get_local_model = function_get_local_model
        self.radii = radii
        self.verbose = verbose

        self.average_f1_score_per_radius, self.std_f1_score_per_radius, \
        self.average_base_line_per_radius, self.average_amount_per_radius = self.evaluate()
        if verbose:
            print(self)

    def get_max_distance(self):
        unbiased_distance_function = Distances.unbiased_distance_function
        max_distance = lambda x:  max([unbiased_distance_function(x, test_instance) for test_instance in self.scaled_test_set])
        return max([max_distance(x) for x in self.scaled_test_set])

    # Compute distances on scaled data
    def get_normalized_distances(self, instance):
        unbiased_distance_function = Distances.unbiased_distance_function
        instance = self.scale(instance)
        distances = map(lambda test_instance: unbiased_distance_function(test_instance, instance), self.scaled_test_set)
        max_distance = float(max(distances))

        return np.array(map(lambda distance: distance/max_distance, distances))

    def get_instances_within_radius(self, normalized_distances, radius):
        indices_filtered = np.where(normalized_distances <= radius)
        return self.test_set[indices_filtered], self.test_predictions[indices_filtered]

    def get_instances_within_radii(self, normalized_distances, radii):
        return [self.get_instances_within_radius(normalized_distances, r) for r in radii]

    def evaluate_instance(self, instance, prediction, radii):
        local_model = self.function_get_local_model(instance, prediction)
        normalized_distances = self.get_normalized_distances(instance)
        instances_within_radii = self.get_instances_within_radii(normalized_distances, radii)

        f1_score = []
        base_line = []
        amount = []
        for instances_within, black_box_predictions in instances_within_radii:
            local_predictions = local_model.get_predictions(instances_within)
            evaluation = EvaluationMetrics(black_box_predictions, local_predictions)
            f1_score.append(evaluation.f1_score)
            base_line_value = Counter(black_box_predictions)[prediction]/float(len(black_box_predictions))
            base_line.append(base_line_value)
            amount.append(len(instances_within))

        stop_index = -1
        for i, base_line_value in zip(range(len(base_line)), base_line):
            if base_line_value <= self.a:
                stop_index = i
                break

        if stop_index == -1:
            print("Stop index is the last index")

        base_line.append(EvaluationMetrics(instances_within_radii[stop_index][1], [prediction]*amount[stop_index]).f1_score)
        f1_score.append(f1_score[stop_index])
        amount.append(amount[stop_index])
        return f1_score, base_line, amount

    def evaluate(self):
        accuracy = []
        base_line = []
        amount = []
        i = 0
        for test_instance, prediction in zip(self.test_set, self.test_predictions):
            i += 1
            if self.verbose:
                print("\t%s/%s" % (i, len(self.test_set)))
            a, b, am = self.evaluate_instance(test_instance, prediction, self.radii)
            accuracy.append(a)
            base_line.append(b)
            amount.append(am)

        average_f1_score_per_radius = np.mean(accuracy, axis=0)
        std_f1_score_per_radius = np.std(accuracy, axis=0)
        average_base_line_per_radius = np.mean(base_line, axis=0)
        average_amount_per_radius = np.mean(amount, axis=0)

        return average_f1_score_per_radius, std_f1_score_per_radius, average_base_line_per_radius, average_amount_per_radius

    def plot(self):
        plt.figure()
        plt.scatter(self.radii, self.average_f1_score_per_radius)
        plt.show()

    def __str__(self):
        return "Radii: %s\nAccuracy: %s\nStd: %s\nAmount: %s" % \
               (self.radii, self.average_f1_score_per_radius, self.std_f1_score_per_radius, self.average_amount_per_radius)

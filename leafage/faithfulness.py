import numpy as np
import matplotlib.pyplot as plt

from local_model import Distances, Neighbourhood
from utils.Evaluate import EvaluationMetrics


class Faithfulness:
    midpoint = "closest_enemy_instance"

    def __init__(self, test_set, test_predictions, function_get_local_model, radii, verbose=False):
        self.scale = lambda x: test_set.pre_process([x], scale=True)[0]
        self.test_set = test_set.feature_vector
        self.scaled_test_set = test_set.scaled_feature_vector
        self.test_predictions = test_predictions
        self.function_get_local_model = function_get_local_model
        self.radii = radii
        self.verbose = verbose

        self.average_accuracy_per_radius, self.std_accuracy_per_radius, self.average_amount_per_radius = self.evaluate()
        if verbose:
            print(self)

    # Compute distances on scaled data
    def get_normalized_distances(self, instance, prediction):
        unbiased_distance_function = Distances.unbiased_distance_function
        instance = self.scale(instance)
        if self.midpoint == "closest_enemy_instance":
            closest_enemy = Neighbourhood.get_closest_enemy_instance(self.scaled_test_set,
                                                                     self.test_predictions,
                                                                     unbiased_distance_function,
                                                                     instance,
                                                                     prediction)
            distances = map(lambda test_instance: unbiased_distance_function(test_instance, closest_enemy), self.scaled_test_set)
        else:
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
        normalized_distances = self.get_normalized_distances(instance, prediction)
        instances_within_radii = self.get_instances_within_radii(normalized_distances, radii)

        evaluation = []
        amount = []
        for instances_within, black_box_predictions in instances_within_radii:
            local_predictions = local_model.get_predictions(instances_within)
            evaluation.append(EvaluationMetrics(black_box_predictions, local_predictions).accuracy)
            amount.append(len(instances_within))
        return evaluation, amount

    def evaluate(self):
        accuracy = []
        amount = []
        i = 0
        for test_instance, prediction in zip(self.test_set, self.test_predictions):
            i += 1
            if self.verbose:
                print("\t%s/%s" % (i, len(self.test_set)))
            e, a = self.evaluate_instance(test_instance, prediction, self.radii)
            accuracy.append(e)
            amount.append(a)

        average_accuracy_per_radius = np.mean(accuracy, axis=0)
        std_accuracy_per_radius = np.std(accuracy, axis=0)
        average_amount_per_radius = np.mean(amount, axis=0)

        return average_accuracy_per_radius, std_accuracy_per_radius, average_amount_per_radius

    def plot(self):
        plt.figure()
        plt.scatter(self.radii, self.average_accuracy_per_radius)
        plt.show()

    def __str__(self):
        return "Radii: %s\nAccuracy: %s\nStd: %s\nAmount: %s" % \
               (self.radii, self.average_accuracy_per_radius, self.std_accuracy_per_radius, self.average_amount_per_radius)

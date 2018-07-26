import numpy as np
import matplotlib.pyplot as plt

from src.leafage.local_model import Distances, Neighbourhood
from src.utils.Evaluate import EvaluationMetrics


class Faithfulness:
    midpoint = "closest_enemy_instance"

    def __init__(self, test_set, test_predictions, function_get_local_model, radii):
        self.test_set = test_set
        self.test_predictions = test_predictions
        self.function_get_local_model = function_get_local_model
        self.radii = radii
        self.accuracy_per_radius, self.amount_per_radius = self.evaluate()
        print(self.amount_per_radius)
        print(self.accuracy_per_radius)

    def get_normalized_distances(self, instance, prediction):
        unbiased_distance_function = Distances.unbiased_distance_function
        if self.midpoint == "closest_enemy_instance":
            closest_enemy = Neighbourhood.get_closest_enemy_instance(self.test_set,
                                                                     self.test_predictions,
                                                                     unbiased_distance_function,
                                                                     instance,
                                                                     prediction)
            distances = map(lambda test_instance: unbiased_distance_function(test_instance, closest_enemy), self.test_set)
        else:
            distances = map(lambda test_instance: unbiased_distance_function(test_instance, instance), self.test_set)

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
        evaluation = []
        amount = []
        i = 0
        for test_instance, prediction in zip(self.test_set, self.test_predictions):
            i += 1
            print("\t%s/%s" % (i, len(self.test_set)))
            e, a = self.evaluate_instance(test_instance, prediction, self.radii)
            evaluation.append(e)
            amount.append(a)

        accuracy_per_radius = []
        amount_per_radius = []
        for i in range(len(self.radii)):
            sum_accuracy = 0
            sum_amount = 0
            for e, a in zip(evaluation, amount):
                sum_accuracy += e[i]
                sum_amount += a[i]
            accuracy_per_radius.append(sum_accuracy/float(len(self.test_set)))
            amount_per_radius.append(sum_amount/float(len(self.test_set)))
        return accuracy_per_radius, amount_per_radius

    def plot(self):
        plt.figure()
        plt.scatter(self.radii, self.accuracy_per_radius)
        plt.show()

    def __str__(self):
        return "Radii: %s\nAccuracy per radius: %s" % (self.accuracy_per_radius, self.radii)

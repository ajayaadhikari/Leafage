from collections import Counter

import numpy as np

from local_model import Distances
from sklearn.metrics import f1_score


class Faithfulness:
    a = 0.95
    radii = np.arange(0.05, 1.05, 0.05)

    def __init__(self, test_set, test_predictions, function_get_local_model, verbose=False):
        self.scale = lambda x: test_set.pre_process([x], scale=True)[0]
        self.test_set = test_set.feature_vector
        self.scaled_test_set = test_set.scaled_feature_vector
        self.test_predictions = test_predictions
        self.function_get_local_model = function_get_local_model
        self.verbose = verbose

        self.f1_score, self.baseline, self.amount = self.evaluate()
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

        for instances_within, black_box_predictions in instances_within_radii:
            local_predictions = local_model.get_predictions(instances_within)
            base_line_value = Counter(black_box_predictions)[prediction]/float(len(black_box_predictions))
            if base_line_value <= self.a:
                f1_score_value = f1_score(black_box_predictions, local_predictions, average="macro")
                base_line = base_line_value
                amount = len(instances_within)
                break

        return f1_score_value, base_line, amount

    def evaluate(self):
        f1 = []
        base_line = []
        amount = []
        i = 0
        for test_instance, prediction in zip(self.test_set, self.test_predictions):
            i += 1
            if self.verbose:
                print("\t%s/%s" % (i, len(self.test_set)))
            f, b, am = self.evaluate_instance(test_instance, prediction, self.radii)
            f1.append(f)
            base_line.append(b)
            amount.append(am)

        return f1, base_line, amount

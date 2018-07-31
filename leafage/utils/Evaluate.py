from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix
import Classifiers
import DimensionalityReduction
from Plots import plot_roc, plot_confusion_matrix


class EvaluationMetrics:
    def __init__(self,
                 real_labels,
                 predicted_labels,
                 predicted_probabilities=None,
                 labels_order_confusion_matrix=None,
                 f1_averaging_strategy=None,
                 sample_weight=None):

        self.real_labels = real_labels
        self.predicted_labels = predicted_labels
        self.predicted_probabilities = predicted_probabilities
        self.labels_order_confusion_matrix = labels_order_confusion_matrix
        self.f1_averaging_strategy = f1_averaging_strategy

        self.confusion_matrix = confusion_matrix(real_labels, predicted_labels, labels=labels_order_confusion_matrix)
        self.accuracy_weighted = accuracy_score(real_labels, predicted_labels, sample_weight=sample_weight)
        self.accuracy = accuracy_score(real_labels, predicted_labels)
        #self.f1_score = f1_score(real_labels, predicted_labels, average=f1_averaging_strategy)

    def plot_ROC(self, machine_learning_method=""):
        plot_roc(self.predicted_labels, self.predicted_probabilities, machine_learning_method)

    def plot_CM(self, class_names, **kwargs):
        plot_confusion_matrix(self.confusion_matrix, class_names, **kwargs)

    def __str__(self):
        return "Accuracy: %s\nConfusion Matrix:\n%s\n" % (self.accuracy, self.confusion_matrix)


class Evaluation:
    def __init__(self, classifier_name, classifier_variables, feature_vector, target_vector, labels_order=None, preprocessing_info=None):
        self.classifier_name = classifier_name
        self.classifier_variables = classifier_variables
        self.feature_vector = feature_vector
        self.target_vector = target_vector
        self.labels_order = labels_order
        self.type_dimensionality_reduction = preprocessing_info
        self.evaluation_metrics = self.evaluate()

    def evaluate(self):

        # Evaluate using k-fold crossvalidation
        k = 5
        k_fold = StratifiedKFold(k)

        print("Performing %s-fold crossvalidation" % k)

        real_labels = []
        predicted_labels = []

        for train, test in k_fold.split(self.feature_vector, self.target_vector):
            training_set = self.feature_vector[train]
            testing_set = self.feature_vector[test]

            if self.type_dimensionality_reduction:
                mapping = DimensionalityReduction.reduce_dimensionality(self.feature_vector[train], self.target_vector[train], self.type_dimensionality_reduction)
                training_set = mapping.transform(training_set)
                testing_set = mapping.transform(testing_set)

            classifier = Classifiers.train(self.classifier_name, training_set, self.target_vector[train], self.classifier_variables)

            real_labels.extend(self.target_vector[test])
            predicted_labels.extend(classifier.predict(testing_set))

        # Get the evaluation metrics
        resulting_metrics = EvaluationMetrics(real_labels, predicted_labels, self.labels_order)
        print("\t\tDone!!")

        return resulting_metrics

    def __str__(self):
        c = "Classifier: %s" % self.classifier_name
        pre = "Type dimensionality reduction: %s" % self.type_dimensionality_reduction
        vars = "Variables: %s" % str(self.classifier_variables)
        labels = "Labels: %s" % self.labels_order
        return "%s\n%s\n%s\n%s\n%s" % (pre, c, vars, labels, str(self.evaluation_metrics))
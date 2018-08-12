from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

possible_classifiers = {"knn": KNeighborsClassifier,
                        "rf": RandomForestClassifier,
                        "mv": VotingClassifier,
                        "gb": GradientBoostingClassifier,
                        "ab": AdaBoostClassifier,
                        "nb_g": GaussianNB,
                        "nb_b": BernoulliNB,
                        "lda": LinearDiscriminantAnalysis,
                        "dt": DecisionTreeClassifier,
                        "svc": SVC,
                        "lr": LogisticRegression,
                        "mlp": MLPClassifier}


def train(name_classifier, features_train, labels_train, variables={}, verbose=False):
    if verbose:
        print("Training ||%s|| with variables %s" % (name_classifier, variables))

    if name_classifier not in possible_classifiers.keys():
        raise(ValueError, "Classifier %s not support choose from: %s" % (name_classifier, possible_classifiers.keys()))
    elif variables is None:
        variables = {}

    classifier = possible_classifiers[name_classifier](**variables)
    classifier.fit(features_train, labels_train)
    return classifier

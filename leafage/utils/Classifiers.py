from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

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
                        "lr": LogisticRegression}


def train(name_classifier, features_train, labels_train, variables):
    print("Training ||%s|| with variables %s" % (name_classifier, variables))
    if name_classifier not in possible_classifiers.keys():
        print("Classifier \"%s\" not support choose from: %s" % (name_classifier, possible_classifiers.keys()))
        return

    if not variables:
        variables = {}

    classifier = possible_classifiers[name_classifier](**variables)
    classifier.fit(features_train, labels_train)
    return classifier

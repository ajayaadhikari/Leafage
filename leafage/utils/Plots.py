import matplotlib.pyplot as plt
import numpy as np
import itertools
from sklearn.metrics import roc_curve, auc
import pandas as pd


###################################################################
# This function prints and plots the confusion matrix.
# Normalization can be applied by setting `normalize=True`.
###################################################################
def plot_confusion_matrix(cm,
                          classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


###################################################################
# Plot the roc of the given scores
###################################################################
def plot_roc(real_labels, probability_labels, classifier_name=""):
    title = 'Receiver operating characteristic for %s' % classifier_name
    if classifier_name is None:
        title = 'Receiver operating characteristic'

    false_positive_rate, true_positive_rate, _ = roc_curve(real_labels, probability_labels, pos_label=1)
    plt.figure()
    lw = 2
    plt.plot(false_positive_rate, true_positive_rate, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % auc(false_positive_rate, true_positive_rate))
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.show()


###################################################################
# Get a histogram of a numerical list
###################################################################
def histogram_numerical(column, column_name):
    plot = pd.Series(column).hist()
    plot.set_title("Histogram of variable %s" % column_name)
    plot.set_xlabel(column_name)
    plot.set_ylabel("count")


###################################################################
# Get a histogram of a categorical list
###################################################################
def histogram_categorical(column, column_name):
    counts = pd.Series(column).value_counts().sort_index()
    plot = counts.plot(kind='bar')
    plot.set_title("Histogram of variable %s" % column_name)
    plot.set_xlabel(column_name)
    plot.set_ylabel("count")
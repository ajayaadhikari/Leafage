from sklearn import decomposition
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


def reduce_dimensionality(feature_vector, labels, method, number_of_dimensions=20):
    print("\tUsing %s" % method)
    if method == "pca":
        pca = decomposition.PCA(n_components=number_of_dimensions)
        return pca.fit(feature_vector)
    elif method == "lda":
        lda = LinearDiscriminantAnalysis(n_components=number_of_dimensions)
        return lda.fit(feature_vector, labels)
    else:
        print("Only lda and pca supported")
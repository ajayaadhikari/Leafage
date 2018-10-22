import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import scikit_posthocs as sp
import numpy as np
from itertools import combinations

df = pd.read_csv("../output/housing/results_user_study_114_filtered.csv")

df["Prediction Estimation binary"] = df["Prediction Estimation"]
df.replace({"Prediction Estimation binary": {"I do not know": "Wrong Estimation"}}, inplace=True)

questions_columns = ["Transparency", "Information Sufficiency",
                     "Competence", "Confidence"]
types_of_explanations = ["No Explanation", "Feature Importance", "Example-based", "Feature and Example-based"]

#df[questions_columns[0]][df["Explanation Type"]==types_of_explanations[0]].hist()

# for column in questions_columns:
#     for explanation_type in types_of_explanations:
#         print("%s, %s" % (column, explanation_type))
#         print("\t(%s,%s)" % stats.shapiro(df[column][df["Explanation Type"]==explanation_type]))


def descriptive_statistics():
    for c in questions_columns:
        print(c)
        for t in types_of_explanations:
            mean_ = np.round(np.mean(df[c][df["Explanation Type"]==t]), 2)
            std_ = np.round(np.std(df[c][df["Explanation Type"]==t]), 2)
            print("\t%s: %s(%s)" % (t, mean_, std_))

    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        cont = pd.crosstab(df["Explanation Type"], df["Prediction Estimation"])
        print("Measured transparency")
        print(cont)

        cont2 = pd.crosstab(df["Explanation Type"], df["Acceptance"])
        print("Acceptance")
        print(cont2)


def post_hoc_test():
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):

        result1 = []
        for column in questions_columns:
            plt.figure()
            print(column)
            plt.figure()
            pc = sp.posthoc_dunn(df, group_col="Explanation Type", val_col=column, p_adjust="bonferroni")
            print(pc)
            result1.append(pc)

            heatmap_args = {'linewidths': 0.25, 'linecolor': '0.5', 'clip_on': False, 'square': True,
                            'cbar_ax_bbox': [0.80, 0.35, 0.04, 0.3]}
            sp.sign_plot(pc, **heatmap_args)
            plt.title(column)
    plt.show()

def kruskal_test():
    for column in questions_columns:
        print("Column: %s" % column)
        print(stats.kruskal(df[column][df["Explanation Type"]==types_of_explanations[0]],
                       df[column][df["Explanation Type"] == types_of_explanations[1]],
                       df[column][df["Explanation Type"] == types_of_explanations[2]],
                       df[column][df["Explanation Type"] == types_of_explanations[3]]))


def chi_square_test_measured_transparency():
    cont = pd.crosstab(df["Explanation Type"], df["Prediction Estimation binary"])
    chi2_stat, p_val, dof, ex = stats.chi2_contingency(cont)
    print("===Chi2 Stat===")
    print(chi2_stat)
    print("\n")
    print("===Degrees of Freedom===")
    print(dof)
    print("\n")
    print("===P-Value===")
    print(p_val)
    print("\n")
    print("===Expected Contingency Table===")
    print(ex)
    print("===Observed Contingency Table===")
    print(cont)


def chi_sqaure_test_acceptance():
    cont = pd.crosstab(df["Explanation Type"], df["Acceptance"])
    chi2_stat, p_val, dof, ex = stats.chi2_contingency(cont)
    print("===Chi2 Stat===")
    print(chi2_stat)
    print("\n")
    print("===Degrees of Freedom===")
    print(dof)
    print("\n")
    print("===P-Value===")
    print(p_val)
    print("\n")
    print("===Expected Contingency Table===")
    print(ex)
    print("===Observed Contingency Table===")
    print(cont)


def post_hoc_measured_transparency():
    combinations_ = list(combinations(types_of_explanations, 2))
    cont = pd.crosstab(df["Explanation Type"], df["Prediction Estimation binary"])
    for i,j in combinations_:
        current_cont = cont.loc[[i,j]]
        chi2_stat, p_val, dof, ex = stats.chi2_contingency(current_cont)
        print("%s vs %s" % (i,j))
        print("\t%s" % np.round(p_val, 3))
        print("\t%s" % np.round((p_val*6), 3))
    a = 5


chi_square_test_measured_transparency()
post_hoc_measured_transparency()

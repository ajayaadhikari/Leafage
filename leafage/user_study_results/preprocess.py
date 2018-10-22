import pandas as pd
import numpy as np

df = pd.read_csv("../data/user_study_results/Housing_market_114.csv", header=None)

#########################################################
# Perceived judgement results
#########################################################
experiment_df = df.iloc[2:, 33:273].fillna(3).astype(int)
num_questions_per_explanation = 6
num_explanations = 40
num_participants = len(df) - 2
questions_columns = ["Estimated Prediction", "Transparency", "Information Sufficiency",
                     "Competence", "Confidence", "Acceptance"]

experiment_values = experiment_df.values.reshape((-1, 6))
processed_experiment_df = pd.DataFrame(experiment_values, columns=questions_columns)
processed_experiment_df["Explanation Type"] = (["Feature Importance"] * 10 + ["Example-based"] * 10 +
                                               ["Feature and Example-based"] * 10 + [
                                                   "No Explanation"] * 10) * num_participants
processed_experiment_df["Participant"] = np.array([[x] * 40 for x in range(num_participants)]).flatten()

#########################################################
# Actual transparency results
#########################################################

correct_solutions = [2,2,1,1,1,1,1,2,2,2,1,1,1,1,1,1,2,2,1,1,1,2,1,1,2,1,1,1,2,1,2,1,1,2,2,2,2,2,2,2]
processed_experiment_df["Correct Prediction"] = correct_solutions * num_participants

measured_transparency = []
for correct_value, estimated_value in zip(processed_experiment_df["Correct Prediction"], processed_experiment_df["Estimated Prediction"]):
    if estimated_value == 3:
        measured_transparency.append("I do not know")
    elif correct_value == estimated_value:
        measured_transparency.append("Correct Estimation")
    else:
        measured_transparency.append("Wrong Estimation")

processed_experiment_df["Is Correct"] = processed_experiment_df["Estimated Prediction"].values == \
                                        processed_experiment_df["Correct Prediction"]
processed_experiment_df["Prediction Estimation"] = measured_transparency
#########################################################
# Demographics
#########################################################
numerical_to_categorical = dict([("Sex", dict([(1, "Male"),
                                               (2, "Female")
                                               ])
                                  ),
                                 ("Highest Level of School", dict([(1, "Less than high school degree"),
                                                                   (2, "High school degree or equivalent (e.g. GED)"),
                                                                   (3, "Some college but no degree"),
                                                                   (4, "Associate degree"),
                                                                   (5, "Bachelor degree"),
                                                                   (6, "Graduate degree")
                                                                   ])
                                  ),
                                 ("Age", dict([(1, "18 to 24"),
                                               (2, "25 to 34"),
                                               (3, "35 to 44"),
                                               (4, "45 to 54"),
                                               (5, "55 to 64"),
                                               (6, "65 to 74"),
                                               (7, "75 or older")
                                               ])
                                  ),
                                 ("Region", dict([(1, "Africa"),
                                                  (2, "Americas"),
                                                  (3, "Asia"),
                                                  (4, "Europe"),
                                                  (5, "Oceania")
                                                  ])
                                  ),
                                 ("Experience with buying a house", dict([(1, "Yes"),
                                                                          (2, "No")
                                                                          ])
                                  ),
                                 ("Level of English", dict([(1, "Bad"),
                                                            (2, "Not very good"),
                                                            (3, "Satisfactory"),
                                                            (4, "Good"),
                                                            (5, "Native/Fluent")
                                                            ])
                                  )])
demographics_df = df.iloc[2:, 9:15].astype(int)
demographics_df.columns = ["Sex", "Age", "Region", "Highest Level of School", "Experience with buying a house",
                           "Level of English"]
for column_name in list(demographics_df):
    demographics_df[column_name] = demographics_df[column_name].map(numerical_to_categorical[column_name])
demographics_df["Participant"] = range(num_participants)

#########################################################
# Score
#########################################################
def get_score(row, typee):
    first_score = row[0] == 1
    second_score = (row[1] == 0) + (row[2] == 0) + (row[3] == 0) + (row[4] == 4) + (row[5] == 5)
    third_score = row[6] == 1
    fourth_score = (row[7] == 1) + (row[8] == 2) + (row[9] == 3) + (row[10] == 0) + (row[11] == 0)

    fifth_score = row[12] == 2
    sixth_score = row[13] == 2

    if typee == 0:
        sum_all = first_score + second_score + third_score + fourth_score + fifth_score + sixth_score
        return sum_all
    else:
        sum_per_question = first_score + (second_score == 5) + third_score + (fourth_score == 5) + fifth_score + sixth_score
        return sum_per_question


indices_score_columns = range(21, 33) + [273, 279]
score_df = df.iloc[2:, indices_score_columns].fillna(0).astype(int)

score_sum_all = [get_score(x, 0) for x in score_df.values.tolist()]
score_per_question = [get_score(x, 1) for x in score_df.values.tolist()]
score_all = np.array([score_sum_all, score_per_question]).transpose()

final_score_df = pd.DataFrame(score_all, columns=["Score sum", "Score per question"])
final_score_df["Participant"] = range(num_participants)
final_score_df["MTurk ID"] = df.iloc[2:, 290].values

final_demographics = final_score_df.merge(demographics_df, on=["Participant"])
final_demographics = final_demographics.loc[final_demographics['Score per question'] >= 5]
print(len(final_demographics))

j = 0
for i in range(len(final_demographics["MTurk ID"].values)):
    j+=1
    print(j)
    print("\t%s, %s" % (final_demographics["MTurk ID"].values[i], final_demographics["Score per question"].values[i]))

from collections import Counter
total = float(len(final_demographics))
for i in ["Sex", "Age", "Region", "Highest Level of School", "Experience with buying a house", "Level of English"]:
    column = final_demographics[i].values
    print(i)
    counter = Counter(column)
    for c in counter.keys():
        print("\t%s: %s (%s%%)" % (c, counter[c], np.round(counter[c]/total*100, 1)))


# final_df = processed_experiment_df.merge(demographics_df, on=["Participant"])
# final_df = final_df.merge(final_score_df, on=["Participant"])
#
# final_df_filtered = final_df.loc[final_df['Score per question'] >= 3]



#final_df.to_csv("../output/housing/results_user_study_114.csv", index=False)
#final_df_filtered.to_csv("../output/housing/results_user_study_114_filtered.csv", index=False)
a = 5
# Thank you for your valuable contribution to the study about housing.
# You scored high on the attention checks.
# Have a nice day :)
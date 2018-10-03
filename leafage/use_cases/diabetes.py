import pandas as pd
from numpy import genfromtxt
import numpy as np

from data import Data


class Diabetes(Data):
    def __init__(self):
        basedir = "../data/"
        feature_names = ["weight", "food_time", "food_cho", "a1_start", "a1_duration_hrs",
                         "a1_hrs_after_food", "a1_end", "a2_start", "a2_duration_hrs",
                         "a2_hrs_after_food", "a2_end", "before_food_hyper", "before_food_hypo",
                         "patient_adolescent", "patient_adult", "patient_child", "food_name_apple",
                         "food_name_banana", "food_name_bread_milk", "food_name_cookie", "food_name_french_fries",
                         "food_name_hamburger_meal", "food_name_juice_nuts", "food_name_milk_crackers",
                         "food_name_peach", "food_name_rice_beans", "food_name_strawberries",
                         "food_name_watermelon", "food_type_meal", "food_type_snack", "a1_type_0",
                         "a1_type_activity", "a1_type_sports", "a1_name_0", "a1_name_bicycling_16kph",
                         "a1_name_chores", "a1_name_dancing", "a1_name_mopping", "a1_name_mountain_climbing",
                         "a1_name_running", "a1_name_swimming", "a1_name_walking", "a2_type_0", "a2_type_activity",
                         "a2_type_sports", "a2_name_0", "a2_name_dancing", "a2_name_skating", "a2_name_walking"]
                         # list(genfromtxt(basedir + 'diabetes_model_feature_names.csv', delimiter=';'))

        data = genfromtxt(basedir + 'diabetes_model_x_train.csv', delimiter=';')
        labels = genfromtxt(basedir + 'diabetes_model_y_train.csv', delimiter=';')
        label_names = ['hyper', 'hypo', 'ok']
        labels = [label_names[np.argmax(v)] for v in labels]

        # data = pd.read_csv('../data/diabetes_model_x_train.csv', na_values="NA", names=feature_names,
        #                    header=None).values
        # labels = pd.read_csv('../data/diabetes_model_x_train.csv', na_values="NA", names=['label'],
        #                      header=None).values

        Data.__init__(self, data, labels, feature_names, name="Diabetes")

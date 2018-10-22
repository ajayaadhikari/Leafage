from data import Data, PreProcess
import pandas as pd
import numpy as np


class HousingDataSet(Data):
    def __init__(self):
        # Read data from file
        original_df = pd.read_csv("../data/housing/train.csv")

        # Get the target vector
        # Make the sale price discrete as ["low", "medium", "high"]
        target_vector = self.split(original_df["SalePrice"].values)
        df = original_df.drop(columns=["SalePrice"])

        # Remove rows with "medium"
        df = df[target_vector!="Medium"]
        target_vector = target_vector[target_vector!="Medium"]

        # Convert null values to string "NA"
        df.fillna("NA", inplace=True)

        # Merge two columns
        df["Bathroom Amount"] = df["BsmtFullBath"] + df["FullBath"]
        df["Toilet Amount"] = df["BsmtHalfBath"] + df["HalfBath"]

        # Remove not needed columns
        df.drop(columns=not_used_columns, inplace=True)

        # Change values of the columns to make them more readable
        for column_name in useful_categorical_columns.keys():
            df[column_name] = df[column_name].map(categorical_values[column_name])
        for column_name in categorical_to_numerical.keys():
            df[column_name] = df[column_name].map(categorical_to_numerical[column_name])

        # Convert to numerical values to int
        df[useful_numerical_columns.keys() + new_columns] = df[useful_numerical_columns.keys() + new_columns].astype(int)

        # Change column names to make them more readable
        df.rename(columns=useful_numerical_columns, inplace=True)
        df.rename(columns=useful_categorical_columns, inplace=True)

        # Only keep final columns
        df = df[final_columns]

        feature_vector = df.values

        # Set the column names as the feature names
        feature_names = list(df)

        Data.__init__(self, feature_vector, target_vector, feature_names, name="Housing")

    @staticmethod
    def split(sale_price):
        result = []
        first_threshold = 150000
        second_threshold = 200000
        low = 0
        high = 0
        for price in sale_price:
            if price <= first_threshold:
                result.append("Low")
                low +=1
            elif price > first_threshold and price <= second_threshold:
                result.append("Medium")
            else:
                result.append("High")
                high +=1
        print(low,high)
        return np.array(result, dtype=object)


not_used_columns = ["Id", "Condition1", "Condition2", "Neighborhood", "RoofMatl", "Exterior1st", "Exterior2nd",
                    "MasVnrType", "MasVnrArea", "BsmtExposure", "BsmtFinType2", "BsmtFinSF2", "BsmtUnfSF",
                    "Heating", "Electrical", "1stFlrSF", "2ndFlrSF", "LowQualFinSF", "BsmtFullBath",
                    "BsmtHalfBath", "BsmtFullBath", "BsmtHalfBath", "FullBath", "HalfBath", "Functional",
                    "GarageYrBlt", "GarageFinish", "GarageArea", "WoodDeckSF", "OpenPorchSF", "EnclosedPorch",
                    "3SsnPorch", "ScreenPorch", "MiscVal", "MoSold", "YrSold", "SaleType", "SaleCondition",
                    "LotFrontage", "BsmtFinType1", "BsmtFinSF1", "MSSubClass"]

final_columns = ["Living Area", "Year Built", "Overall Quality(1-10)", "Bathroom Amount", "Bedroom Amount"]

new_columns = ["Bathroom Amount", "Toilet Amount"]
useful_numerical_columns = dict([("LotArea", "Total Area"),
                                 ("OverallQual", "Overall Quality(1-10)"),
                                 ("OverallCond", "Overall Condition(1-10)"),
                                 ("YearBuilt", "Year Built"),
                                 ("YearRemodAdd", "Year Renovation"),
                                 ("ExterQual", "Quality Exterior(1-5)"),
                                 ("ExterCond", "Condition Exterior(1-5)"),
                                 ("TotalBsmtSF", "Basement Area"),
                                 ("HeatingQC", "Heating Condition"),
                                 ("GrLivArea", "Living Area"),
                                 ("BedroomAbvGr", "Bedroom Amount"),
                                 ("KitchenAbvGr", "Amount Kitchens"),
                                 ("KitchenQual", "Kitchen Quality(1-5)"),
                                 ("TotRmsAbvGrd", "Total Rooms"),
                                 ("Fireplaces", "Amount Fireplaces"),
                                 ("GarageCars", "Garage Capacity"),
                                 ("PoolArea", "PoolArea"),
                                 ("HouseStyle", "Floor Amount"),
                                 ])
categorical_to_numerical = dict([("ExterQual",   dict([("Ex", 5),
                                                       ("Gd", 4),
                                                       ("TA", 3),
                                                       ("Fa", 2),
                                                       ("Po", 1)
                                                       ])
                                  ),
                                 ("ExterCond",  dict([("Ex", 5),
                                                      ("Gd", 4),
                                                      ("TA", 3),
                                                      ("Fa", 2),
                                                      ("Po", 1)
                                                      ])
                                  ),
                                 ("HeatingQC", dict([("Ex", 5),
                                                     ("Gd", 4),
                                                     ("TA", 3),
                                                     ("Fa", 2),
                                                     ("Po", 1)
                                                     ])
                                  ),
                                 ("KitchenQual", dict([("Ex", 5),
                                                       ("Gd", 4),
                                                       ("TA", 3),
                                                       ("Fa", 2),
                                                       ("Po", 1)
                                                     ])
                                  ),
                                 ("HouseStyle", dict([("1Story", 1),
                                                      ("1.5Fin", 2),
                                                      ("1.5Unf", 2),
                                                      ("2Story", 2),
                                                      ("2.5Fin", 3),
                                                      ("2.5Unf", 3),
                                                      ("SFoyer", 2),
                                                      ("SLvl", 2)
                                                      ])
                                  ),
                                 ])

useful_categorical_columns = dict([("Street",       "Road Type"),
                                   ("Alley",        "Alley Type"),
                                   ("LotShape",     "Shape Property"),
                                   ("LandContour",  "Flatness"),
                                   ("Utilities",    "Utilities Available"),
                                   ("LotConfig",    "Lot Configuration"),
                                   ("LandSlope",    "Land Slope"),
                                   ("BldgType",     "Type"),
                                   ("RoofStyle",    "Roof Stype"),
                                   ("Foundation",   "Type of Foundation"),
                                   ("BsmtQual",     "Basement Height"),
                                   ("BsmtCond",     "Basement Condition"),
                                   ("CentralAir",   "Central Air Conditioning"),
                                   ("FireplaceQu",  "Fireplace Quality"),
                                   ("GarageType",   "Garage Type"),
                                   ("GarageQual",   "Garage Quality"),
                                   ("GarageCond",   "Garage Condition"),
                                   ("PoolQC",       "Pool Quality"),
                                   ("PavedDrive",   "Driveway"),
                                   ("Fence",        "Fence Quality"),
                                   ("MiscFeature",  "Extras"),
                                   ("MSZoning",     "Neighbourhood")
                                   ])
categorical_values = dict([("Street",       dict([("Grvl", "Gravel"),
                                                  ("Pave", "Paved")
                                                  ])
                            ),
                           ("Alley",        dict([("Grvl", "Gravel"),
                                                  ("Pave", "Paved"),
                                                  ("NA", "No Alley")
                                                  ])
                            ),
                           ("LotShape",     dict([("Reg", "Regular"),
                                                  ("IR1", "Slightly Irregular"),
                                                  ("IR2", "Moderately Irregular"),
                                                  ("IR3", "Irregular")
                                                  ])
                            ),
                           ("LandContour",  dict([("Lvl", "Flat"),
                                                  ("Bnk", "Rise from street"),
                                                  ("HLS", "Rise side to side"),
                                                  ("Low", "Not flat")
                                                  ])
                            ),
                           ("Utilities",    dict([("AllPub", "Elec., Gas and Water"),
                                                  ("NoSewr", "Elec., Gas and Water"),
                                                  ("NoSeWa", "Elec. and Gas only"),
                                                  ("ELO", "Elec. only"),
                                                  ])
                            ),
                           ("LotConfig",    dict([("Inside", "Inside"),
                                                  ("Corner", "Corner"),
                                                  ("CulDSac", "Inside"),
                                                  ("FR2", "Corner"),
                                                  ("FR3", "Corner")
                                                  ])
                            ),
                           ("LandSlope",    dict([("Gtl", "Gentle Slope"),
                                                  ("Mod", "Moderate Slope"),
                                                  ("Sev", "Severe Slope")
                                                  ])
                            ),
                           ("BldgType",     dict([("1Fam", "Single-Family"),
                                                  ("2fmCon", "Two apartments"),
                                                  ("Duplex", "Duplex"),
                                                  ("TwnhsE", "Townhouse"),
                                                  ("Twnhs", "Townhouse")
                                                  ])
                            ),
                           ("RoofStyle",    dict([("Flat", "Flat"),
                                                  ("Gable", "Not Flat"),
                                                  ("Gambrel", "Not Flat"),
                                                  ("Hip", "Not Flat"),
                                                  ("Mansard", "Not Flat"),
                                                  ("Shed", "Not Flat")
                                                  ])
                            ),
                           ("Foundation",   dict([("BrkTil", "Brik and Tile"),
                                                  ("CBlock", "Cinder Block"),
                                                  ("PConc", "Concrete"),
                                                  ("Slab", "Concrete"),
                                                  ("Stone", "Stone"),
                                                  ("Wood", "Wood")
                                                  ])
                            ),
                           ("BsmtQual",     dict([("Ex", "Excellent"),
                                                  ("Gd", "Good"),
                                                  ("TA", "Average"),
                                                  ("Fa", "Fair"),
                                                  ("Po", "Poor"),
                                                  ("NA", "No Basement")
                                                  ])
                            ),
                           ("BsmtCond",     dict([("Ex", "Excellent"),
                                                  ("Gd", "Good"),
                                                  ("TA", "Average"),
                                                  ("Fa", "Fair"),
                                                  ("Po", "Poor"),
                                                  ("NA", "No Basement")
                                                  ])
                            ),
                           ("CentralAir",   dict([("N", "No"),
                                                  ("Y", "Yes")
                                                  ])
                            ),
                           ("FireplaceQu",  dict([("Ex", "Excellent"),
                                                  ("Gd", "Good"),
                                                  ("TA", "Average"),
                                                  ("Fa", "Fair"),
                                                  ("Po", "Poor"),
                                                  ("NA", "No Fireplace")
                                                  ])
                            ),
                           ("GarageType",    dict([("2Types", "Attached to house"),
                                                  ("Attchd", "Attached to house"),
                                                  ("Basment", "Basement"),
                                                  ("BuiltIn", "Built-in"),
                                                  ("CarPort", "Detached from house"),
                                                  ("Detchd", "Detached from house"),
                                                  ("NA", "No Garage")
                                                  ])
                            ),
                           ("GarageQual", dict([("Ex", "Excellent"),
                                                  ("Gd", "Good"),
                                                  ("TA", "Average"),
                                                  ("Fa", "Fair"),
                                                  ("Po", "Poor"),
                                                  ("NA", "No Garage")
                                                  ])
                            ),
                           ("GarageCond", dict([("Ex", "Excellent"),
                                                     ("Gd", "Good"),
                                                     ("TA", "Average"),
                                                     ("Fa", "Fair"),
                                                     ("Po", "Poor"),
                                                     ("NA", "No Garage")
                                                     ])
                            ),
                           ("PavedDrive",       dict([("Y", "Paved"),
                                                      ("P", "Partially Paved"),
                                                      ("N", "Not Paved")
                                                      ])
                            ),
                           ("Fence",            dict([("GdPrv", "Excellent"),
                                                      ("MnPrv", "Good"),
                                                      ("GdWo", "Average"),
                                                      ("MnWw", "Fair"),
                                                      ("NA", "No Fence")
                                                      ])
                            ),
                           ("MiscFeature",      dict([("Elev", "Elevator"),
                                                      ("Gar2", "Second Garage"),
                                                      ("Othr", "None"),
                                                      ("Shed", "Shed"),
                                                      ("TenC", "Tennis Court"),
                                                      ("NA", "None")
                                                      ])
                            ),
                           ("PoolQC",           dict([("Ex", "Excellent"),
                                                      ("Gd", "Good"),
                                                      ("TA", "Average"),
                                                      ("Fa", "Fair"),
                                                      ("NA", "No Pool")
                                                      ])
                            ),
                           ("MSZoning",         dict([("A", "Agriculture"),
                                                      ("C (all)", "Town Center"),
                                                      ("FV", "House on Water"),
                                                      ("I", "Industrial"),
                                                      ("RH", "Residential High Density"),
                                                      ("RL", "Residential Low Density"),
                                                      ("RP", "Residential Low Density"),
                                                      ("RM", "Residential Medium Density")
                                                      ])
                            )
                           ])

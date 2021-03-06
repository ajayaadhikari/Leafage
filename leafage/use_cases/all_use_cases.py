from handwritten_digits import DigitsDataset
from iris import IrisDataSet
from adult import Adult
from housing import HousingDataSet
from breast_cancer import BreastCancerDataset
from diabetes import Diabetes
from wine import WineDataset
from abalone import Abalone
from bank_note import BankNote

all_data_sets = {"iris": IrisDataSet,
                 "digits": DigitsDataset,
                 "adult": Adult,
                 "housing": HousingDataSet,
                 "wine": WineDataset,
                 "breast_cancer": BreastCancerDataset,
                 "abalone": Abalone,
                 "bank_note": BankNote,
                 "diabetes": Diabetes}

if __name__ == "__main__":
    a = HousingDataSet()
    b = 2
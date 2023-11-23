import pickle
import numpy as np
import pandas as pd

pipe = pickle.load(open("Regression.pkl", "rb"))


def make_prediction():
    input = pd.DataFrame(
        [
            [
                "RL",
                8450,
                "Pave",
                "Norm",
                "2Story",
                5,
                "Gable",
                "CompShg",
                "PConc",
                "Gd",
                "GasA",
                "Ex",
                "Y",
                "SBrkr",
                2,
                "Gd",
                8,
                "Typ",
                "Attchd",
            ]
        ],
        columns=[
            "MSZoning",
            "LotArea",
            "Street",
            "Condition1",
            "HouseStyle",
            "OverallCond",
            "RoofStyle",
            "RoofMatl",
            "Foundation",
            "BsmtQual",
            "Heating",
            "HeatingQC",
            "CentralAir",
            "Electrical",
            "FullBath",
            "KitchenQual",
            "TotRmsAbvGrd",
            "Functional",
            "GarageType",
        ],
    )
    prediction = pipe.predict(input)[0]
    print(prediction)
    return prediction.tolist()


# prediction = make_prediction(loaded_model, temperature, temp_min, temp_max,humidity, rainfall, visibility, windspeed_min, windspeed_max)
prediction2 = make_prediction()
x = [prediction2]
print(x)

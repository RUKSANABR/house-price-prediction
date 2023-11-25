from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import requests
import pickle
from datetime import datetime
import jsonpickle
from sklearn.preprocessing import StandardScaler


app = Flask(__name__)


@app.route("/api", methods=["GET"])
def returnAscii():
    d = {}
    inputchr = str(request.args["query"])
    msZoning = str(inputchr).split("|")[0]
    lotArea = str(inputchr).split("|")[1]
    street = str(inputchr).split("|")[2]
    conditional1 = str(inputchr).split("|")[3]
    houseStyle = str(inputchr).split("|")[4]
    overallCond = str(inputchr).split("|")[5]
    roofStyle = str(inputchr).split("|")[6]
    roofMat1 = str(inputchr).split("|")[7]
    foundation = str(inputchr).split("|")[8]
    bsmtQual = str(inputchr).split("|")[9]
    heating = str(inputchr).split("|")[10]
    heatingQC = str(inputchr).split("|")[11]
    centralAir = str(inputchr).split("|")[12]
    electrical = str(inputchr).split("|")[13]
    fullBath = str(inputchr).split("|")[14]
    kitchenQual = str(inputchr).split("|")[15]
    totRmsAbvGrd = str(inputchr).split("|")[16]
    functional = str(inputchr).split("|")[17]
    garageType = str(inputchr).split("|")[18]
    print(msZoning)
    print(lotArea)

    # load the model from disk

    pipe = pickle.load(open("Regression.pkl", "rb"))

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
    return jsonify({"price": prediction})


if __name__ == "main":
    app.run()

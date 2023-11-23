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
    latitude = str(inputchr).split("|")[0]
    longitude = str(inputchr).split("|")[1]
    print(latitude)
    print(longitude)

    # load the model from disk
    loaded_model = pickle.load(open("housePrice.pkl", "rb"))

    def make_prediction(
        loaded_model,
        Area,
        No_of_Bedrooms,
        Resale,
        MaintenanceStaff,
        Gymnasium,
        SwimmingPool,
        LandscapedGardens,
        JoggingTrack,
        RainWaterHarvesting,
        IndoorGames,
        ShoppingMall,
        Intercom,
        SportsFacility,
        ATM,
        ClubHouse,
        School,
        Security,
        PowerBackup,
        CarParking,
        StaffQuarter,
        Cafeteria,
        MultipurposeRoom,
        Hospital,
        WashingMachine,
        Gasconnection,
        AC,
        Wifi,
        playarea,
        LiftAvailable,
        BED,
        VaastuCompliant,
        Microwave,
        TV,
        Sofa,
        Wardrobe,
    ):
        input_data = np.array(
            [
                Area,
                No_of_Bedrooms,
                Resale,
                MaintenanceStaff,
                Gymnasium,
                SwimmingPool,
                LandscapedGardens,
                JoggingTrack,
                RainWaterHarvesting,
                IndoorGames,
                ShoppingMall,
                Intercom,
                SportsFacility,
                ATM,
                ClubHouse,
                School,
                Security,
                PowerBackup,
                CarParking,
                StaffQuarter,
                Cafeteria,
                MultipurposeRoom,
                Hospital,
                WashingMachine,
                Gasconnection,
                AC,
                Wifi,
                playarea,
                LiftAvailable,
                BED,
                VaastuCompliant,
                Microwave,
                TV,
                Sofa,
                Wardrobe,
            ]
        ).reshape(1, -1)
        sc = StandardScaler()
        x = sc.fit_transform(input_data)
        prediction = loaded_model.predict(input_data)
        return prediction.tolist()

    # prediction = make_prediction(loaded_model, temperature, temp_min, temp_max,humidity, rainfall, visibility, windspeed_min, windspeed_max)
    prediction2 = make_prediction(
        loaded_model,
        1126,
        2,
        0,
        0,
        1,
        1,
        1,
        0,
        1,
        1,
        0,
        0,
        0,
        0,
        1,
        0,
        0,
        1,
        0,
        0,
        0,
        1,
        0,
        0,
        0,
        0,
        0,
        1,
        1,
        0,
        0,
        0,
        0,
        0,
        0,
    )
    x = [prediction2]
    print(x)
    return jsonpickle.encode(x)


if __name__ == "main":
    app.run()

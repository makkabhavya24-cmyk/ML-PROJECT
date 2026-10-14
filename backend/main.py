from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd


# --------------------------------------------------
# CREATE FASTAPI APP
# --------------------------------------------------

app = FastAPI(
    title="Rainfall-Based Crop Yield Prediction",
    description="Predict crop yield using rainfall, weather, soil and agricultural data.",
    version="3.0"
)


# --------------------------------------------------
# CORS
# --------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------
# LOAD MODEL AND ENCODERS
# --------------------------------------------------

model = joblib.load("rainfall_yield_model.pkl")

state_encoder = joblib.load("state_encoder.pkl")
crop_encoder = joblib.load("crop_encoder.pkl")
season_encoder = joblib.load("season_encoder.pkl")


# --------------------------------------------------
# INPUT DATA MODEL
# --------------------------------------------------

class PredictionInput(BaseModel):

    year: int
    state: str
    crop: str
    season: str

    area: float

    temperature: float
    annual_rainfall: float
    humidity: float

    N: float
    P: float
    K: float
    pH: float

    fertilizer: float
    pesticide: float


# --------------------------------------------------
# HOME
# --------------------------------------------------

@app.get("/")
def home():

    return {
        "message": "Rainfall-Based Crop Yield Prediction API is running!",
        "version": "3.0",
        "features": 14
    }


# --------------------------------------------------
# GET DROPDOWN OPTIONS
# --------------------------------------------------

@app.get("/options")
def get_options():

    return {
        "states": state_encoder.classes_.tolist(),
        "crops": crop_encoder.classes_.tolist(),
        "seasons": season_encoder.classes_.tolist()
    }


# --------------------------------------------------
# PREDICT CROP YIELD
# --------------------------------------------------

@app.post("/predict")
def predict_yield(data: PredictionInput):

    try:

        # ------------------------------------------
        # ENCODE CATEGORICAL VALUES
        # ------------------------------------------

        state_encoded = state_encoder.transform(
            [data.state]
        )[0]

        crop_encoded = crop_encoder.transform(
            [data.crop]
        )[0]

        season_encoded = season_encoder.transform(
            [data.season]
        )[0]


        # ------------------------------------------
        # CREATE MODEL INPUT
        # ------------------------------------------

        input_data = pd.DataFrame([{

            "Year": data.year,

            "State_encoded": state_encoded,

            "Crop_encoded": crop_encoded,

            "Season_encoded": season_encoded,

            "Area": data.area,

            "Temperature": data.temperature,

            "Annual_Rainfall": data.annual_rainfall,

            "Humidity": data.humidity,

            "N": data.N,

            "P": data.P,

            "K": data.K,

            "pH": data.pH,

            "Fertilizer": data.fertilizer,

            "Pesticide": data.pesticide

        }])


        # ------------------------------------------
        # PREDICTION
        # ------------------------------------------

        prediction = model.predict(input_data)[0]


        # ------------------------------------------
        # RESPONSE
        # ------------------------------------------

        return {

            "status": "success",

            "predicted_yield": round(
                float(prediction),
                2
            ),

            "unit": "yield units per hectare"

        }


    except ValueError as e:

        return {

            "status": "error",

            "message": f"Invalid state, crop or season: {str(e)}"

        }


    except Exception as e:

        return {

            "status": "error",

            "message": str(e)

        }
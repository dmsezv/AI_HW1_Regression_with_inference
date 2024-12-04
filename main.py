import uvicorn
import io
from typing import List

from schemas.Car import Car, Cars

from services.formatter import get_clean_data_frame
from services.preprocessor import preprocess_input
from services.model_loader import load_models

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.encoders import jsonable_encoder

from contextlib import asynccontextmanager

import pandas as pd


models = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    medians, ohe, scaler, ridge_model = load_models()

    models["medians"] = medians
    models["ohe"] = ohe
    models["scaler"] = scaler
    models["ridge_model"] = ridge_model
    yield
    models.clear()


app = FastAPI(lifespan=lifespan)


@app.get("/")
async def read_root():
    return {"Car Price Predictor": "Using Ridge linear model for prediction"}


@app.post("/predict_price")
async def predict_price(car: Car) -> float:
    try:
        input_data = pd.DataFrame([jsonable_encoder(car)])
        return float(make_prediction(input_data)[0])
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error: {str(e)}")


@app.post("/predict_prices")
async def predict_prices(cars: Cars) -> List[float]:
    input_data = pd.DataFrame(jsonable_encoder(cars.list))
    return make_prediction(input_data).tolist()


@app.post("/predict_prices_csv")
async def predict_prices_csv(csv_file: UploadFile = File(...)):
    try:
        data = await csv_file.read()
        df_raw = pd.read_csv(io.StringIO(data.decode("utf-8")))
        df_raw["selling_price"] = make_prediction(df_raw)

        output = io.StringIO()
        df_raw.to_csv(output, index=False)
        output.seek(0)

        response = StreamingResponse(
            output,
            media_type="text/csv"
        )
        response.headers["Content-Disposition"] = "attachment; filename=predicted_cars_prices.csv"

        return response
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error: {str(e)}")


def make_prediction(df_raw):
    df = df_raw.copy()
    df_clean = get_clean_data_frame(df, models["medians"])
    processed_data = preprocess_input(df_clean, models)
    return models["ridge_model"].predict(processed_data)


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)

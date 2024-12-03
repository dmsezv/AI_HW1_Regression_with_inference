import uvicorn
import io

from schemas.Car import Car, Cars

from services.formatter import get_clean_data_frame
from services.preprocessor import preprocess_input
from services.model_loader import load_models

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse

from contextlib import asynccontextmanager

import pandas as pd

models = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    ohe, scaler, ridge_model = load_models()

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
async def predict_price(car: Car):
    try:
        input_data = pd.DataFrame([car.dict()])
        processed_data = preprocess_input(input_data, models)
        prediction = models["ridge_model"].predict(processed_data)

        return {"predicted_price": prediction[0]}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error: {str(e)}")


@app.post("/predict_prices_from_raw_csv")
async def predict_prices_csv(csv_file: UploadFile = File(...)):
    try:
        data = await csv_file.read()

        df_clean = get_clean_data_frame(io.StringIO(data.decode("utf-8")))
        df_preprocessed = preprocess_input(df_clean, models)

        df_raw = pd.read_csv(io.StringIO(data.decode("utf-8")))
        df_raw["selling_price"] = models["ridge_model"].predict(df_preprocessed)

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


# async def pydantic_model_to_df(model_instance):
#     pd.DataFrame([jsonable_encoder(model_instance)])


# async def ml_lifespan_manager(app: FastAPI):
#     ml_models["grid_search_ridge"] = grid_search_ridge
#     yield
#     ml_models.clear()


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)

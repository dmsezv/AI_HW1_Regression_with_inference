import pickle

OHE_PATH = "ml_models/one_hot_encoder.pkl"
MODEL_PATH = "ml_models/grid_search_ridge_model.pkl"
SCALER_PATH = "ml_models/scaler.pkl"


def load_models():
    try:
        with open(OHE_PATH, "rb") as file:
            ohe = pickle.load(file)
        with open(MODEL_PATH, "rb") as file:
            ridge_model = pickle.load(file)
        with open(SCALER_PATH, "rb") as file:
            scaler = pickle.load(file)
        return ohe, scaler, ridge_model
    except Exception as e:
        raise RuntimeError(f"[ERROR]:\nexp load models:{e}\n")

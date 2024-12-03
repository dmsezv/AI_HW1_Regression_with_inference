import pandas as pd

num_cols = ["year", "km_driven", "nm_torque", "bhp_max_power", "cc_engine", "kmpl_mileage"]
cat_cols = ["fuel", "seller_type", "transmission", "owner", "seats", "mark", "model"]


def preprocess_input(df, models) -> pd.DataFrame:
    df_scaled = pd.DataFrame(
        models["scaler"].transform(df[num_cols]),
        columns=num_cols,
        index=df.index
    )

    df_encoded = pd.DataFrame(
        models["ohe"].transform(df[cat_cols]),
        columns=models["ohe"].get_feature_names_out(cat_cols),
        index=df.index
    )

    df_transformed = pd.concat([df_scaled, df_encoded], axis=1)
    df_transformed = df_transformed.reindex(columns=models["ridge_model"].feature_names_in_, fill_value=0)

    return df_transformed

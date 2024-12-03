import pandas as pd
import re


def get_clean_data_frame(file):
    df = pd.read_csv(file)

    df["bhp_max_power"] = df["max_power"].apply(convert_to_num)
    df["kmpl_mileage"] = df["mileage"].apply(convert_to_num)
    df["cc_engine"] = df["engine"].apply(convert_to_num)

    df["cc_engine"] = df["cc_engine"].fillna(0).astype(int)
    df["seats"] = df["seats"].fillna(0).astype(int)

    df[["mark", "model"]] = df["name"].apply(convert_name)

    df["nm_torque"] = df.apply(lambda row: convert_to_nm(row["torque"], row["name"]), axis=1)

    if "selling_price" in df.columns:
        df = df.drop(columns=["selling_price"])

    df = df.drop(columns=["max_power", "mileage", "engine", "torque", "name"])

    df = fill_nan_median(df)

    return df


def fill_nan_median(df):
    na_columns = df.columns[df.isna().any()].tolist()
    medians = df[na_columns].median()
    df[na_columns] = df[na_columns].fillna(medians)
    return df


def convert_to_nm(torque, name):
    if pd.isna(torque):
        return None

    torque = torque.lower()

    try:
        # явная опечатка(789Nm), проверил характеристики этого авто в интернете
        if name == "Maruti Zen D":
            return 78.9

        if "kgm" in torque and "nm" in torque:
            return float(torque.split("nm")[0])

        torque_value = float(torque.split("@")[0].split("nm")[0].split("kgm")[0].split()[0])

        if "kgm" in torque:
            # добавил условие которое исправляет явные опечатки в единицах измерения
            if torque_value < 100:
                torque_value *= 9.80665
        return torque_value
    except Exception as ex:
        if torque == "110(11.2)@ 4800":
            return float(torque.split("(")[0])
        else:
            print(f"[ERROR]:\nvalue: {torque}\ne: {ex}\n")
            return None


def convert_name(value):
    result = value.split(" ")

    mark = result[0]
    model = result[1]
    return pd.Series([mark, model], index=["mark", "model"])


def convert_to_num(value):
    if pd.isna(value):
        return None

    try:
        num_value = float(value.split(" ")[0])
        if num_value == 0.0:
            return None
        else:
            return num_value
    except Exception as e:
        print(f"[ERROR]:\nval: {value}\nex: {e}\n")
        return None
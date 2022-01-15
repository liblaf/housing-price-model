import numpy as np
import pandas as pd
from os import path
from ..search_poi_amap import CATEGORIES


def read_housing() -> pd.DataFrame:
    current_dir = path.dirname(path.abspath(__file__))
    target_filename = path.join(current_dir, "shanghai_housing_prepared.csv")
    if path.exists(target_filename):
        return pd.read_csv(target_filename)

    def to_float(raw):
        try:
            return float(raw)
        except:
            return np.nan

    housing = pd.read_csv(
        path.join(current_dir, "shanghai_housing.csv"),
        header=0,
        converters={"均价": to_float, "经度_WGS1984坐标": to_float, "纬度_WGS1984坐标": to_float},
    )
    housing = housing[["均价", "经度_WGS1984坐标", "纬度_WGS1984坐标"]]
    housing.rename(
        columns={"均价": "price", "经度_WGS1984坐标": "lng", "纬度_WGS1984坐标": "lat"},
        inplace=True,
    )
    housing.dropna(axis="index", subset=["price", "lng", "lat"], inplace=True)
    return housing


def read_poi() -> pd.DataFrame:
    current_dir = path.dirname(path.abspath(__file__))
    target_filename = path.join(current_dir, "shanghai_poi_prepared.csv")
    if path.exists(target_filename):
        return pd.read_csv(target_filename)
    poi = pd.DataFrame()
    for type in CATEGORIES:
        new_poi = pd.read_csv(path.join(current_dir, f"shanghai_{type}.csv"))
        new_poi["type"] = type
        new_poi["score"] = np.exp(np.linspace(start=3, stop=-3, num=len(new_poi)))
        poi = poi.append(
            new_poi[["name", "type", "location", "score"]], ignore_index=True
        )
        pass
    lng = np.zeros(len(poi))
    lat = np.zeros(len(poi))
    for i, row in poi.iterrows():
        lng[i], lat[i] = eval(row["location"])
    poi["lng"] = lng
    poi["lat"] = lat
    del poi["location"]
    return poi


def read():
    return read_housing(), read_poi()

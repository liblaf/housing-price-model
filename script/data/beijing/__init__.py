import numpy as np
import pandas as pd
from os import path
from ..search_poi_amap import CATEGORIES


def read_housing() -> pd.DataFrame:
    current_dir = path.dirname(path.abspath(__file__))
    target_filename = path.join(current_dir, "beijing_housing_prepared.csv")
    if path.exists(target_filename):
        return pd.read_csv(target_filename)
    housing = pd.read_csv(
        path.join(current_dir, "beijing_housing.csv"),
        header=0,
        dtype={"price": np.float64, "Lng": np.float64, "Lat": np.float64},
    )
    housing = housing[["price", "Lng", "Lat"]]
    housing.rename(columns={"Lng": "lng", "Lat": "lat"}, inplace=True)
    housing.dropna(subset=["price", "lng", "lat"], inplace=True)
    housing.to_csv(target_filename)
    return housing


def read_poi() -> pd.DataFrame:
    current_dir = path.dirname(path.abspath(__file__))
    target_filename = path.join(current_dir, "beijing_poi_prepared.csv")
    if path.exists(target_filename):
        return pd.read_csv(target_filename)
    poi = pd.DataFrame()
    for type in CATEGORIES:
        new_poi = pd.read_csv(path.join(current_dir, f"beijing_{type}.csv"))
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
    poi.to_csv(target_filename)
    return poi


def read():
    return read_housing(), read_poi()

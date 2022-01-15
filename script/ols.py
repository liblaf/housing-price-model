from distutils.command.check import HAS_DOCUTILS
from os import path
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import data


def lng_lat_to_x_y(lng, lat, origin_lng, origin_lat) -> tuple:
    EARTH_RADIUS = 6371.000
    x = np.deg2rad(lng - origin_lng) * EARTH_RADIUS * np.cos(np.deg2rad(lat))
    y = np.deg2rad(lat - origin_lat) * EARTH_RADIUS
    return x, y


def calculate_characteristics(housing: pd.DataFrame, poi: pd.DataFrame) -> tuple:
    categories = poi["type"].unique()
    housing[categories] = 0
    for i, row in poi.iterrows():
        if i % 1000 == 0:
            print(f"progress: {int(i * 100 / len(poi))}%")
        dist = np.sqrt((housing["x"] - row["x"]) ** 2 + (housing["y"] - row["y"]) ** 2)
        score = np.exp(-(dist / row["score"]))
        score[np.logical_not(np.isfinite(score))] = 0
        housing[row["type"]] += score
    return housing, poi, categories


def prepare(housing: pd.DataFrame, poi: pd.DataFrame, city: str) -> tuple:
    print("transforming coordinates ...")
    origin_lng = housing["lng"].mean()
    origin_lat = housing["lat"].mean()
    housing["x"], housing["y"] = lng_lat_to_x_y(
        housing["lng"], housing["lat"], origin_lng, origin_lat
    )
    poi["x"], poi["y"] = lng_lat_to_x_y(poi["lng"], poi["lat"], origin_lng, origin_lat)
    del housing["lng"], housing["lat"], poi["lng"], poi["lat"]
    print("calculating characteristics ...")
    current_dir = path.dirname(path.abspath(__file__))
    target_filename = path.join(current_dir, f"{city}_prepared_for_fit.csv")
    if path.exists(target_filename):
        housing = pd.read_csv(target_filename)
        categories = poi["type"].unique()
    else:
        housing, poi, categories = calculate_characteristics(housing, poi)
        for category in categories:
            housing[category] = (
                housing[category] - housing[category].mean()
            ) / housing[category].std()
        housing.to_csv(target_filename)
    print("data prepared for fit:")
    print(f"    number of samples:         {len(housing)}")
    print(f"    number of POIs:            {len(poi)}")
    print(f"    number of characteristics: {len(categories)}")
    return housing, poi, categories


def main(city: str):
    housing, poi = data.import_data(city)
    housing, poi, categories = prepare(housing, poi, city)
    # housing = housing.sample(frac=1, ignore_index=True)
    housing.sort_values(by="price", ascending=True, inplace=True, ignore_index=True)
    plt.figure()
    plt.gca().set_aspect(1)
    plt.scatter(x=housing["x"], y=housing["y"], s=1, c=housing["price"])
    colorbar = plt.colorbar()
    colorbar.set_label("Price")
    plt.savefig(f"{city}_housing.png")

    # poi = poi.sample(frac=1, ignore_index=True)
    poi.sort_values(by="score", ascending=True, inplace=True, ignore_index=True)
    plt.figure()
    plt.gca().set_aspect(1)
    plt.scatter(x=poi["x"], y=poi["y"], s=1, c=poi["score"])
    colorbar = plt.colorbar()
    colorbar.set_label("Impact Scope")
    plt.savefig(f"{city}_poi.png")

    print("fitting data ...")
    model = smf.ols(f"price ~ {' + '.join(categories)}", data=housing)
    results = model.fit()
    print(results.summary(), file=open(f"{city}_ols.txt", mode="w"))

    print("preparing data for prediction ...")
    n = 1000
    x_fit = np.linspace(housing["x"].min(), housing["x"].max(), n)
    y_fit = np.linspace(housing["y"].min(), housing["y"].max(), n)
    x_fit, y_fit = np.meshgrid(x_fit, y_fit)
    data_fit = pd.DataFrame()
    data_fit["x"] = x_fit.flatten()
    data_fit["y"] = y_fit.flatten()
    data_fit, _, _ = calculate_characteristics(data_fit, poi)
    for category in categories:
        data_fit[category] = (data_fit[category] - housing[category].mean()) / housing[
            category
        ].std()
    price_fit = results.predict(data_fit).to_numpy()
    price_fit = np.reshape(price_fit, newshape=(n, n))
    plt.figure()
    plt.gca().set_aspect(1)
    plt.imshow(price_fit, extent=(x_fit.min(), x_fit.max(), y_fit.min(), y_fit.max()))
    colorbar = plt.colorbar()
    colorbar.set_label("Price")
    plt.savefig(f"{city}_predict.png")

    params = results.params
    del params["Intercept"]
    params.sort_values(ascending=True, inplace=True)
    plt.figure(figsize=(8, 4.8))
    plt.barh(range(len(params)), params)
    plt.yticks(range(len(params)), params.index)
    plt.xlabel("Impact Factor")
    plt.subplots_adjust(left=0.4)
    plt.savefig(f"{city}_impact.png")

    print("All done!")


if __name__ == "__main__":
    main(sys.argv[1])

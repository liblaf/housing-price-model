from . import beijing
from . import shanghai
from .search_poi_amap import CATEGORIES


def import_data(city: str):
    print(f"importing data from {city} ...")
    if city == "beijing":
        return beijing.read()
    elif city == "shanghai":
        return shanghai.read()


if __name__ == "__main__":
    housing, poi = import_data("Beijing")
    print(poi)

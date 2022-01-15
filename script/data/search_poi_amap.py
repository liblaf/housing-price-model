import pandas as pd
import requests

CITIES = ["beijing", "shanghai"]
CITIES_ZH = ["北京", "上海"]
CATEGORIES = [
    "AutoService",
    "AutoDealers",
    "AutoRepair",
    "MotorcycleService",
    "FoodBeverages",
    "Shopping",
    "DailyLifeService",
    "SportsRecreation",
    "MedicalService",
    "AccommodationService",
    "TouristAttraction",
    "CommercialHouse",
    "GovernmentalOrganizationSocialGroup",
    "ScienceCultureEducationService",
    # "CultureEducation",
    "TransportationService",
    "FinanceInsuranceService",
    "Enterprises",
    "RoadFurniture",
    "PlaceNameAddress",
    "PublicFacility",
    "IncidentsandEvents",
    "Indoorfacilities",
    "PassFacilities",
]
CATEGORIES_ZH = [
    "汽车服务",
    "汽车销售",
    "汽车维修",
    "摩托车服务",
    "餐饮服务",
    "购物服务",
    "生活服务",
    "体育休闲服务",
    "医疗保健服务",
    "住宿服务",
    "风景名胜",
    "商务住宅",
    "政府机构及社会团体",
    "科教文化服务",
    "交通设施服务",
    "金融保险服务",
    "公司企业",
    "道路附属设施",
    "地名地址信息",
    "公共设施",
    "事件活动",
    "室内设施",
    "通行设施",
]


def requests_get(url: str, params: dict):
    ok = False
    while ok == False:
        try:
            output = requests.get(url, params)
            ok = output.ok
        except:
            ok = False
    return output


def search_poi(city: str, types: str):
    page = 1
    pois = pd.DataFrame()
    no_data = False
    while no_data == False:
        output = requests_get(
            url="https://restapi.amap.com/v3/place/text",
            params={
                "key": "4c81ae5627426784dab88a5c5642aca0",
                "types": types,
                "city": city,
                "citylimit": True,
                "offset": 25,
                "page": page,
            },
        ).json()
        no_data = int(output["count"]) == 0
        if no_data:
            break
        page += 1
        pois = pois.append(pd.DataFrame(output["pois"]), ignore_index=True)
    return pois


if __name__ == "__main__":
    for i in range(len(CITIES)):
        for j in range(len(CATEGORIES)):
            pois = search_poi(CITIES_ZH[i], CATEGORIES_ZH[j])
            pois.to_csv(f"{CITIES[i]}/{CITIES[i]}_{CATEGORIES[j]}.csv")
            print(f"number of {CATEGORIES[j]} in {CITIES[i]}: {len(pois)}")

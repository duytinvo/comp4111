import requests
import json
from bs4 import BeautifulSoup


def get_distance(start, stop):
    api = "yourkeyhere"
    url = "https://maps.googleapis.com/maps/api/distancematrix/json?units=imperial&origins=" \
          + start + "&destinations=" + stop + "&key=" + api
    link = requests.get(url)
    json_loc = link.json()
    distance = json_loc['rows'][0]['elements'][0]['distance']['text']
    distance = int(''.join([d for d in distance if d.isdigit() == True]))
    return distance


cities = """
New York
Los Angeles
Chicago
Houston
Phoenix
Philadelphia
San Antonio
San Diego
Dallas
San Jose
Austin
Jacksonville
Fort Worth
Columbus
Charlotte
San Francisco
Indianapolis
Seattle
Denver
Washington DC
Boston
El Paso
Nashville
Detroit
Oklahoma City
"""
cities = [c for c in cities.split('\n') if c != '']

edges = []
dist_dict = {c: {} for c in cities}
for idx_1 in range(0, len(cities) - 1):
    for idx_2 in range(idx_1 + 1, len(cities)):
        city_a = cities[idx_1]
        city_b = cities[idx_2]
        dist = get_distance(city_a, city_b)
        dist_dict[city_a][city_b] = dist
        edges.append((city_a, city_b, dist))

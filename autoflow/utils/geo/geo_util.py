#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
cities geo data from https://github.com/echarts-maps/echarts-china-cities-js
regions center geo from https://github.com/boyan01/ChinaRegionDistrict
"""

from scipy.spatial.ckdtree import cKDTree
import os
import json
from shapely.geometry import shape, Point


def change_geojson(geojson_path=None):
    this_dir = os.path.dirname(os.path.realpath(__file__))
    if geojson_path is None:
        geojson_path = os.path.join(this_dir, '../../../works/regional_words_bag/geojson/shape-only')
    res = []
    for file_name in os.listdir(geojson_path):
        with open(os.path.join(geojson_path, file_name), 'r') as f:
            g = json.load(f)
        res.append(g)
    geojson_file = os.path.join(this_dir, 'geojson.json')
    with open(geojson_file, 'w') as f:
        json.dump(res, f)


def get_city_geo_dict() -> dict:
    res = dict()
    this_dir = os.path.dirname(os.path.realpath(__file__))
    region_json_path = os.path.join(this_dir, 'region.json')
    with open(region_json_path, 'r') as f:
        country = json.load(f)
    for province in country['districts']:
        if province['name'] in {'重庆市', '北京市', '上海市', '天津市', '香港特别行政区', '澳门特别行政区'}:
            cities = [{'name': province['name'][:2], 'center': province['center']}]
        else:
            cities = province['districts']
        for city in cities:
            city_name = city['name'].strip('市')
            res[city_name] = [city['center']['longitude'], city['center']['latitude']]
    return res


class GeoUtil:

    def __init__(self):
        this_dir = os.path.dirname(os.path.realpath(__file__))
        self.geojson_path = os.path.join(this_dir, 'geojson.json')
        self.region_json_path = os.path.join(this_dir, 'region.json')
        self.regions: list = []
        self.kd: cKDTree
        self.city_polygon_dict: dict = {}

        self._build_kdtree()
        self._build_polygon()
        # self._fix_test()

    def _build_kdtree(self):
        with open(self.region_json_path, 'r') as f:
            country = json.load(f)
        points = []
        for province in country['districts']:
            if province['name'] in {'重庆市', '北京市', '上海市', '天津市', '香港特别行政区', '澳门特别行政区'}:
                cities = [{'name': province['name'][:2], 'center': province['center']}]
            else:
                cities = province['districts']
            for city in cities:
                city_name = city['name'].strip('市')
                self.regions.append((city_name, province['name']))
                points.append([city['center']['longitude'], city['center']['latitude']])
        self.kd = cKDTree(points)

    def _build_polygon(self):
        with open(self.geojson_path, 'r') as f:
            gs = json.load(f)
        for g in gs:
            for ft in g['features']:
                self.city_polygon_dict[ft['properties']['name']] = shape(ft['geometry'])

    def _fix_test(self):
        d = set()
        for c, p in self.regions:
            d.add(c)
            if c not in self.city_polygon_dict:
                print(c, p)
        for c in self.city_polygon_dict:
            if c not in d:
                print(c)

    def get_city(self, lng, lat):
        if not (73.5 <= lng <= 135 and 4 <= lat <= 53.5):
            return 'foreign', 'foreign'
        p = Point(lng, lat)
        _, ii = self.kd.query([lng, lat], k=3)
        city, province = '', ''
        for i in ii:
            city, province = self.regions[i]
            if self.city_polygon_dict[city].contains(p):
                return city, province
        return city, province


if __name__ == '__main__':
    change_geojson()

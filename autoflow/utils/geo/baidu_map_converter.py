import math

# https://github.com/wandergis/coordTransform_py/blob/master/coordTransform_utils.py

xu = 6370996.81
Sp = [1.289059486E7, 8362377.87, 5591021, 3481989.83, 1678043.12, 0]
Hj = [75, 60, 45, 30, 15, 0]

a = 6378245.0  # 长半轴
ee = 0.00669342162296594323  # 偏心率平方

Au = [[1.410526172116255e-8, 0.00000898305509648872, -1.9939833816331, 200.9824383106796, -187.2403703815547,
       91.6087516669843, -23.38765649603339, 2.57121317296198, -0.03801003308653, 17337981.2],
      [-7.435856389565537e-9, 0.000008983055097726239, -0.78625201886289, 96.32687599759846, -1.85204757529826,
       -59.36935905485877, 47.40033549296737, -16.50741931063887, 2.28786674699375, 10260144.86],
      [-3.030883460898826e-8, 0.00000898305509983578, 0.30071316287616, 59.74293618442277, 7.357984074871,
       -25.38371002664745, 13.45380521110908, -3.29883767235584, 0.32710905363475, 6856817.37],
      [-1.981981304930552e-8, 0.000008983055099779535, 0.03278182852591, 40.31678527705744, 0.65659298677277,
       -4.44255534477492, 0.85341911805263, 0.12923347998204, -0.04625736007561, 4482777.06],
      [3.09191371068437e-9, 0.000008983055096812155, 0.00006995724062, 23.10934304144901, -0.00023663490511,
       -0.6321817810242, -0.00663494467273, 0.03430082397953, -0.00466043876332, 2555164.4],
      [2.890871144776878e-9, 0.000008983055095805407, -3.068298e-8, 7.47137025468032, -0.00000353937994,
       -0.02145144861037, -0.00001234426596, 0.00010322952773, -0.00000323890364, 826088.5]]
Qp = [[-0.0015702102444, 111320.7020616939, 1704480524535203, -10338987376042340, 26112667856603880, -35149669176653700,
       26595700718403920, -10725012454188240, 1800819912950474, 82.5],
      [0.0008277824516172526, 111320.7020463578, 647795574.6671607, -4082003173.641316, 10774905663.51142,
       -15171875531.51559, 12053065338.62167, -5124939663.577472, 913311935.9512032, 67.5],
      [0.00337398766765, 111320.7020202162, 4481351.045890365, -23393751.19931662, 79682215.47186455,
       -115964993.2797253, 97236711.15602145, -43661946.33752821, 8477230.501135234, 52.5],
      [0.00220636496208, 111320.7020209128, 51751.86112841131, 3796837.749470245, 992013.7397791013, -1221952.21711287,
       1340652.697009075, -620943.6990984312, 144416.9293806241, 37.5],
      [-0.0003441963504368392, 111320.7020576856, 278.2353980772752, 2485758.690035394, 6070.750963243378,
       54821.18345352118, 9540.606633304236, -2710.55326746645, 1405.483844121726, 22.5],
      [-0.0003218135878613132, 111320.7020701615, 0.00369383431289, 823725.6402795718, 0.46104986909093,
       2351.343141331292, 1.58060784298199, 8.77738589078284, 0.37238884252424, 7.45]]


def Yr(x, y, b):
    if b is not None:
        c = b[0] + b[1] * abs(x)
        d = abs(y) / b[9]
        d = b[2] + b[3] * d + b[4] * d * d + b[5] * d * d * d + b[6] * d * d * d * d + b[7] * d * d * d * d * d + b[
            8] * d * d * d * d * d * d
        lon = c * (-1 if 0 > x else 1)
        lat = d * (-1 if 0 > y else 1)
        return round(lon, 6), round(lat, 6)
    else:
        return None


def Mercator_to_BD09(lng, lat):
    x, y = abs(lng), abs(lat)
    for d in range(len(Sp)):
        if y >= Sp[d]:
            c = Au[d]
            break
    return Yr(x, y, c)


def BD09_to_GCJ02(bd_lng, bd_lat):
    x = bd_lng - 0.0065
    y = bd_lat - 0.006
    x_pi = 3.14159265358979324 * 3000.0 / 180.0
    z = math.sqrt(x * x + y * y) - 0.00002 * math.sin(y * x_pi)
    theta = math.atan2(y, x) - 0.000003 * math.cos(x * x_pi)
    gcj_lng = z * math.cos(theta)
    gcj_lat = z * math.sin(theta)
    return gcj_lng, gcj_lat


def Mercator2GCJ02(lng, lat):
    lng1, lat1 = Mercator_to_BD09(lng, lat)
    return BD09_to_GCJ02(lng1, lat1)


#########
# WGS84 #
#########

def out_of_china(lng, lat):
    """
    判断是否在国内，不在国内不做偏移
    :param lng:
    :param lat:
    :return:
    """
    return not (lng > 73.66 and lng < 135.05 and lat > 3.86 and lat < 53.55)


def _transformlat(lng, lat):
    ret = -100.0 + 2.0 * lng + 3.0 * lat + 0.2 * lat * lat + \
          0.1 * lng * lat + 0.2 * math.sqrt(math.fabs(lng))
    ret += (20.0 * math.sin(6.0 * lng * math.pi) + 20.0 *
            math.sin(2.0 * lng * math.pi)) * 2.0 / 3.0
    ret += (20.0 * math.sin(lat * math.pi) + 40.0 *
            math.sin(lat / 3.0 * math.pi)) * 2.0 / 3.0
    ret += (160.0 * math.sin(lat / 12.0 * math.pi) + 320 *
            math.sin(lat * math.pi / 30.0)) * 2.0 / 3.0
    return ret


def _transformlng(lng, lat):
    ret = 300.0 + lng + 2.0 * lat + 0.1 * lng * lng + \
          0.1 * lng * lat + 0.1 * math.sqrt(math.fabs(lng))
    ret += (20.0 * math.sin(6.0 * lng * math.pi) + 20.0 *
            math.sin(2.0 * lng * math.pi)) * 2.0 / 3.0
    ret += (20.0 * math.sin(lng * math.pi) + 40.0 *
            math.sin(lng / 3.0 * math.pi)) * 2.0 / 3.0
    ret += (150.0 * math.sin(lng / 12.0 * math.pi) + 300.0 *
            math.sin(lng / 30.0 * math.pi)) * 2.0 / 3.0
    return ret


def GCJ02_to_WGS84(lng, lat):
    """
    GCJ02(火星坐标系)转GPS84
    :param lng:火星坐标系的经度
    :param lat:火星坐标系纬度
    :return:
    """
    if out_of_china(lng, lat):
        return [lng, lat]
    dlat = _transformlat(lng - 105.0, lat - 35.0)
    dlng = _transformlng(lng - 105.0, lat - 35.0)
    radlat = lat / 180.0 * math.pi
    magic = math.sin(radlat)
    magic = 1 - ee * magic * magic
    sqrtmagic = math.sqrt(magic)
    dlat = (dlat * 180.0) / ((a * (1 - ee)) / (magic * sqrtmagic) * math.pi)
    dlng = (dlng * 180.0) / (a / sqrtmagic * math.cos(radlat) * math.pi)
    mglat = lat + dlat
    mglng = lng + dlng
    return [lng * 2 - mglng, lat * 2 - mglat]


def BD09_to_WGS84(bd_lon, bd_lat):
    lon, lat = BD09_to_GCJ02(bd_lon, bd_lat)
    return GCJ02_to_WGS84(lon, lat)


def Mercator2WGS84(lng, lat):
    lng1, lat1 = Mercator_to_BD09(lng, lat)
    return BD09_to_WGS84(lng1, lat1)


############
# distance #
############


def _rad(d):
    return d * math.pi / 180.0


def _deg(a):
    return a * 180.0 / math.pi


def distance(lng1, lat1, lng2, lat2):
    radLat1 = _rad(lat1)
    radLat2 = _rad(lat2)
    a = radLat1 - radLat2
    b = _rad(lng1) - _rad(lng2)
    s = 2 * math.asin(math.sqrt(math.pow(math.sin(a / 2), 2) +
                                math.cos(radLat1) * math.cos(radLat2) * math.pow(math.sin(b / 2), 2)))
    s = s * xu
    return s


def get_bound_by_distance(lng, lat, d):
    rad_lng = _rad(lng)
    rad_lat = _rad(lat)
    lng_c = 2*math.asin(math.sin(d/2/xu)/math.cos(rad_lat))
    lat_c = math.asin(2*math.sin(d/2/xu))
    return _deg(rad_lng-lng_c), _deg(rad_lng+lng_c), _deg(rad_lat-lat_c), _deg(rad_lat+lat_c)


def get_lng_fix_ratio_fn(real_dist=1000, p=5):
    dd = math.sin(real_dist/2/xu)
    deg = 180.0 / math.pi
    deg2 = 360.0 / math.pi
    d = deg * math.asin(2*dd)  # 0.00899322 -- 1000
    map = dict()

    def fix_lng_lat(lng, lat):
        lat = round(lat, p)
        if lat not in map:
            lng_d = deg2 * math.asin(dd / math.cos(lat / deg))
            map[lat] = d / lng_d
        return round(lng * map[lat], p), lat

    return d, fix_lng_lat


#####################################
# api.map.baidu.com/lbsapi/getpoint #
#####################################


def parse_single(content):
    bd_lng, bd_lat = Mercator_to_BD09(content['x'] / 100, content['y'] / 100)
    gc_lng, gc_lat = BD09_to_GCJ02(bd_lng, bd_lat)
    wg_lng, wg_lat = GCJ02_to_WGS84(gc_lng, gc_lat)
    lng_lat = list(map(lambda x: str(x), [bd_lng, bd_lat, gc_lng, gc_lat, wg_lng, wg_lat]))
    alias = ';'.join(map(lambda x: str(x), content.get('alias', [])))
    return ','.join([content['name']] + lng_lat + [alias, content.get('std_tag', '')])


if __name__ == '__main__':
    # print(get_bound_by_distance(123.4675, 41.7910, 11))
    print(distance(111.94, 42.89, 111.941, 42.89))
    # 1 -- 11120
    # 2 -- 1112
    # 3 -- 111.2 103.2 81.47
    # 4 -- 11.12
    # 5 -- 1.112
    # 6 -- 0.1112

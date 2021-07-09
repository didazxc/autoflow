# coding=utf-8

from pyspark import SparkContext
from pyspark.sql import SparkSession, functions as F, Row
from pyspark.sql.types import StringType
import hashlib
from typing import Dict, Any, Iterable
from functools import reduce


def load_spark_map(spark: SparkSession, file_path: str, separator: str = '\t'):
    sc = spark.sparkContext

    def flat_map_func(row):
        arr = row.strip().split(separator)
        if len(arr) == 2:
            return [(arr[0].strip(), arr[1].strip())]
        return []

    return sc.textFile(file_path).flatMap(flat_map_func).collectAsMap()


def load_map(file_path: str, separator: str = '\t', encoding: str = 'utf8'):
    m = {}
    with open(file_path, encoding=encoding) as f:
        for line in f:
            arr = line.split(separator, 1)
            m[arr[0].strip()] = arr[1].strip()
    return m


def md5(user_id: str):
    if (len(user_id) == 32):
        return user_id
    else:
        m = hashlib.md5(user_id.encode('utf8'))
        return m.hexdigest()


udf_md5 = F.udf(md5, StringType())


def combine_imei_aid(simei: str, aid: str, oem: int):
    if simei and len(simei) == 32:
        return simei
    elif oem == 2:
        return aid
    else:
        return md5(aid.strip().lower())


udf_get_userid = F.udf(combine_imei_aid, StringType())


def combine_dict_inplace(a: Dict[Any, int or float], b: Dict[Any, int or float]):
    for k in b:
        a[k] = a.get(k, 0) + b[k]
    return a


def combine_dicts(dicts: Iterable[Dict[Any, int or float]]):
    res = dict()
    reduce(combine_dict_inplace, dicts, res)
    return res


def is_hdfs_exist(sc: SparkContext, path: str):
    fs = sc._jvm.org.apache.hadoop.fs.FileSystem.get(sc._jsc.hadoopConfiguration())
    return fs.exists(sc._jvm.org.apache.hadoop.fs.Path(path))

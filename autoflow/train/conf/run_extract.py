from pyspark.sql import SparkSession
import json

APP_LIST_SIZE = 1000

spark = SparkSession.builder.master('local[*]').appName('test') \
    .config('spark.driver.memory', '20g') \
    .config('spark.driver.maxResultSize', '20g') \
    .config('spark.kryoserializer.buffer.max', '1g') \
    .enableHiveSupport() \
    .getOrCreate()


def extract_app_table():
    applist = spark.sql('select applist from dida.ks_label_weekfull where applist is not null') \
        .rdd.flatMap(lambda row: map(lambda app: (app.split(':')[0], 1), row.applist)).reduceByKey(
        lambda a, b: a + b).sortBy(lambda x: -x[1]).map(lambda x: f'{x[0]}\n').take(APP_LIST_SIZE)
    with open('applist.txt', 'w', encoding='utf8') as f:
        f.writelines(applist)


def extract_userprofile_table():
    arr = spark.sql('select userprofile from dida.ks_label_weekfull where userprofile is not null') \
        .rdd.flatMap(lambda r: map(lambda k: (k, {r.userprofile[k] if r.userprofile[k] is not None else '-1'}), r.userprofile.asDict()))\
        .reduceByKey(lambda a, b: a | b)\
        .mapValues(lambda x: {v: i for i, v in enumerate(sorted(list(x)))}).collect()
    res = {k[0]: k[1] for k in arr}
    f = open('userprofile.json', 'w', encoding='utf8')
    json.dump(res, f)
    f.close()


if __name__ == '__main__':
    extract_userprofile_table()

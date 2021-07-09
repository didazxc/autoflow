# import findspark
#
# findspark.init()

from pyspark.sql import SparkSession
from autoflow.utils.geo.baidu_map_converter import get_lng_fix_ratio_fn
from scipy.spatial import cKDTree


def run():
    spark = SparkSession.builder.enableHiveSupport().getOrCreate()
    data = spark.read.format("csv").option("header", "true").load('xxz/location.csv').select('gc_lng', 'gc_lat') \
        .rdd.map(lambda x: (float(x[0]), float(x[1]))).collect()
    dataBC = spark.sparkContext.broadcast(data)

    def map_partitions_fn(rows):
        d, fn = get_lng_fix_ratio_fn(200, p=5)
        # 500 -- 4357184
        # 300 -- 2641363
        # 200 -- 1456985  only school 708589
        # 150 -- 1000944
        # 100 -- 569416
        # 80 -- 406965
        # 70 -- 336074
        # 50 -- 191986
        # 20 -- 36788
        kd_tree = cKDTree(list(map(lambda x: fn(x[0], x[1]), dataBC.value)))
        for row in rows:
            for geo in row['geo']:
                try:
                    arr = list(map(lambda x: round(float(x), 5), geo.split(':', 1)[1].split(',', 1)))
                    x = fn(arr[0], arr[1])
                except Exception:
                    continue
                else:
                    s = kd_tree.query(x)
                    if s[0] <= d:
                        yield (row['aid'],)
                    break

    df = spark.sql("select aid,geo from userinput.merge_orc where date=20210507 and (fr not like '%oem%') and length(fr)>1")
    #print("count distinct aid of common version: ", df.select("aid").distinct().count())
    #print("has geo: ", df.where('size(geo)>0').select("aid").distinct().count())
    rdd = df.rdd.mapPartitions(map_partitions_fn)
    spark.createDataFrame(rdd, 'aid:string').distinct().write.mode('overwrite').text('xxz/aid_in_college.txt')
    print("in college: ", spark.read.text('xxz/aid_in_college.txt').count())


if __name__ == '__main__':
    run()

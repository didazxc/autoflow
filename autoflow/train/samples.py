from pyspark.sql import SparkSession
from autoflow.datas import merge_orc_df as datas_merge_orc_df
from autoflow.samples import SamplesRuleExtractor
import datetime
import heapq
from autoflow.utils.table_config import HiveTable


def merge_orc_df_rule_before_reduce(tag_rules, end_date_str, days=7, only_public_version=False, data_max_length=1000):
    start_date_str = (datetime.datetime.strptime(end_date_str, '%Y%m%d') + datetime.timedelta(days=1 - days)).strftime(
        '%Y%m%d')
    if only_public_version:
        sql = f"select imei as userid,data,size(data) data_size,date from {HiveTable.origin_lines_table} where date between {start_date_str} and {end_date_str} and (fr not like '%oem%') and length(fr)>1"
    else:
        sql = f"select imei as userid,data,size(data) data_size,date from {HiveTable.origin_lines_table} where date between {start_date_str} and {end_date_str}"
    spark = SparkSession.builder.enableHiveSupport().getOrCreate()

    sr = SamplesRuleExtractor(tag_rules=tag_rules)
    df = sr.transform(spark.sql(sql))

    max_chars_bc = spark.sparkContext.broadcast(data_max_length)

    def reduce_fn(a: dict, b: dict):
        max_chars = max_chars_bc.value
        user1, user2 = (a, b) if (a['date'] >= b['date']) else (b, a)
        if user1['data_size'] == max_chars:
            pass
        elif user1['data_size'] > max_chars:
            user1['data'] = user1['data'][:max_chars]
            user1['data_size'] = max_chars
        else:
            data_size = user1['data_size'] + user2['data_size']
            if data_size <= max_chars:
                user1['data'] = user2['data'] + user1['data']
                user1['data_size'] = user1['data_size'] + user2['data_size']
            else:
                user1['data'] = user2['data'][:max_chars - user2['data_size']] + user1['data']
                user1['data_size'] = max_chars
        user1_tags = user1['tags']
        user2_tags = user2['tags']
        for k in range(len(user2_tags)):
            user1_tags[k] += user2_tags[k]
        return user1

    def flat_map_fn(row):
        top2 = heapq.nlargest(2, row['tags'])
        if top2[0] > top2[1]:
            row['label'] = float(row['tags'].index(top2[0]))
            yield row

    rdd = df.rdd.map(lambda row: (row['userid'], row.asDict())).reduceByKey(reduce_fn).values().flatMap(flat_map_fn)

    return spark.createDataFrame(rdd)


def merge_orc_df(tag_rules, end_date_str, days=7, only_public_version=False, data_max_length=1000):
    df = datas_merge_orc_df(days, end_date_str, only_public_version, data_max_length)
    sr = SamplesRuleExtractor(tag_rules=tag_rules)
    return sr.transform(df)

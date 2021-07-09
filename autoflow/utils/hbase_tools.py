import json
from pyspark.sql import DataFrame

zkquorum = ""
znode = "/hbase2"


def get_hbase_rdd(sc, table: str, key_name: str = 'userid'):
    hbaseconf = {"hbase.zookeeper.quorum": zkquorum, "zookeeper.znode.parent": znode,
                 "hbase.mapreduce.inputtable": table}
    keyConv = "org.apache.spark.examples.pythonconverters.ImmutableBytesWritableToStringConverter"
    valueConv = "org.apache.spark.examples.pythonconverters.HBaseResultToStringConverter"
    hbase_rdd = sc.newAPIHadoopRDD("org.apache.hadoop.hbase.mapreduce.TableInputFormat",
                                   "org.apache.hadoop.hbase.io.ImmutableBytesWritable",
                                   "org.apache.hadoop.hbase.client.Result",
                                   keyConverter=keyConv,
                                   valueConverter=valueConv,
                                   conf=hbaseconf)

    def map_func(row):
        res = {i['qualifier']: i['value'] for i in (json.loads(k) for k in row[1].split('\n'))}
        res[key_name] = row[0]
        return res

    return hbase_rdd.map(map_func)


def save_hbase_df(df: DataFrame, table: str, key_col='userid', cf='t'):
    '''maybe need kinit first'''
    keyConv = "org.apache.spark.examples.pythonconverters.StringToImmutableBytesWritableConverter"
    valueConv = "org.apache.spark.examples.pythonconverters.StringListToPutConverter"
    conf = {"hbase.zookeeper.quorum": zkquorum, "zookeeper.znode.parent": znode,
            "hbase.mapred.outputtable": table,
            "mapreduce.outputformat.class": "org.apache.hadoop.hbase.mapreduce.TableOutputFormat",
            "mapreduce.job.output.key.class": "org.apache.hadoop.hbase.io.ImmutableBytesWritable",
            "hbase.client.keyvalue.maxsize": "524288000",
            "mapreduce.job.output.value.class": "org.apache.hadoop.io.Writable"}

    def flat_map_func(row):
        d = row.asDict()
        row_key = d[key_col]
        return [(row_key, [row_key, cf, k,
                           v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)])
                for k, v in d.items() if k != key_col and v]

    rdd = df.rdd.flatMap(flat_map_func)
    rdd.saveAsNewAPIHadoopDataset(conf=conf, keyConverter=keyConv, valueConverter=valueConv)

from pyspark.sql import SparkSession
from autoflow.utils.table_config import HiveTable
from .run import trainProcess


if __name__ == '__main__':

    spark = SparkSession.builder.enableHiveSupport().getOrCreate()

    # aid date

    # sample from merge_orc
    df = spark.sql(f"select simei as userid,data,date from {HiveTable.origin_lines_table} where date={trainProcess.dirs.sample_date_str} and (fr not like '%oem%') and length(fr)>1")
    df.transform(df).write.mode('overwrite').orc(trainProcess.dirs.sample_corpus_orc_path)

    trainProcess.sample_balance(trainProcess.dirs.sample_corpus_orc_path,
                                trainProcess.dirs.sample_corpus_balance_orc_path, 1000000)

    # sample from apps
    corpus_df = spark.read.orc(trainProcess.dirs.sample_corpus_balance_orc_path).select('userid', 'label', 'tags', 'data')
    apps_df = spark.sql(f"select aid as userid,applist from {HiveTable.origin_applist_table} where date={trainProcess.dirs.sample_date_str} and (fr not like '%oem%') and length(fr)>1")
    apps_df.join(corpus_df, 'userid').select('userid', 'label', 'applist', 'data') \
        .write.mode('overwrite').orc(trainProcess.dirs.sample_applist_orc_path)
    trainProcess.sample_balance(trainProcess.dirs.sample_applist_orc_path,
                                trainProcess.dirs.sample_applist_balance_orc_path, 1000000)

    # convert to sample file
    trainProcess.train_convert()

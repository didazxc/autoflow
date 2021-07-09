from pyspark.sql import SparkSession
from autoflow.train.conf.dir_conf import ModelDirConf
from autoflow.datas.merge_orc import merge_orc_df
from autoflow.datas.applist_multi_orc import applist_multi_orc_df
from autoflow.train.transformers.converter import DataConverterTransformer
from autoflow.train.utils.pipeline_tools import MmlShim
from pyspark.sql.types import StructType, StructField, StringType, LongType, ArrayType, DoubleType
from pyspark.ml.linalg import VectorUDT
from pyspark.ml import PipelineModel
from autoflow.utils.hbase_tools import save_userprofile
from autoflow.utils.functions import is_hdfs_exist
from autoflow.samples import SamplesRuleExtractor


class TrainProcess:
    converted_schema = StructType([
        StructField('userid', StringType(), False),
        StructField('data', ArrayType(LongType()), False),
        StructField('applist', VectorUDT(), False),
        StructField('appfreq', VectorUDT(), False),
        StructField('appdur', VectorUDT(), False),
        StructField('appsys', VectorUDT(), False),
        StructField('date', StringType(), False),
    ])
    training_schema = StructType([
        StructField('userid', StringType(), True),
        StructField('label', DoubleType(), True),
        StructField('data', ArrayType(LongType()), True),
        StructField('applist', VectorUDT(), True),
        StructField('appfreq', VectorUDT(), True),
        StructField('appdur', VectorUDT(), True),
        StructField('appsys', VectorUDT(), True),
    ])

    def __init__(self, name, sample_date_str, predict_date_str=None, version='latest', model_name='train_model', days=7,
                 data_max_length=1000, only_public_version=False):
        assert sample_date_str is not None
        self.name = name
        self.version = version
        self.dirs = ModelDirConf(name, version, model_name, sample_date_str, predict_date_str)
        # other attr
        self.days = days
        self.data_max_length = data_max_length
        self.only_public_version = only_public_version

    # sample

    def sample_corpus(self, tag_rules):
        df = merge_orc_df(days=self.days, end_date_str=self.dirs.sample_date_str,
                          only_public_version=self.only_public_version, data_max_length=self.data_max_length)
        sr = SamplesRuleExtractor(tag_rules=tag_rules)
        sr.transform(df).write.mode('overwrite').orc(self.dirs.sample_corpus_orc_path)

    @staticmethod
    def sample_balance(in_orc, out_orc, cnt=100000):
        spark = SparkSession.builder.getOrCreate()
        df = spark.read.orc(in_orc)
        m = df.groupBy('label').count().rdd.map(lambda row: (str(row['label']), row['count'])).collectAsMap()
        df.where('label=0.0').sample(False, float(cnt) / m['0.0']) \
            .union(df.where('label=1.0').sample(False, float(cnt) / m['1.0'])) \
            .write.mode('overwrite').orc(out_orc)

        spark.read.orc(out_orc).groupBy('label').count().show()

    def sample_apps(self):
        spark = SparkSession.builder.enableHiveSupport().getOrCreate()
        corpus_df = spark.read.orc(self.dirs.sample_corpus_balance_orc_path).select('userid', 'label', 'tags', 'data')
        apps_df = applist_multi_orc_df(days=1, end_date_str=self.dirs.sample_date_str)
        apps_df.join(corpus_df, 'userid').select('userid', 'label', 'applist', 'data') \
            .write.mode('overwrite').orc(self.dirs.sample_applist_orc_path)

    def sample(self, tag_rules):
        self.sample_corpus(tag_rules)
        self.sample_balance(self.dirs.sample_corpus_orc_path, self.dirs.sample_corpus_balance_orc_path, 1000000)
        self.sample_apps()
        self.sample_balance(self.dirs.sample_applist_orc_path, self.dirs.sample_applist_balance_orc_path, 200000)

    # train

    def train_convert(self):
        spark = SparkSession.builder.getOrCreate()
        df = spark.read.orc(self.dirs.sample_applist_balance_orc_path)
        converter = DataConverterTransformer()
        df = converter.transform(df)
        df.printSchema()
        df.show()
        train_df, test_df = df.randomSplit([0.7, 0.3], 17)
        train_df.write.mode('overwrite').orc(self.dirs.train_orc_path)
        test_df.write.mode('overwrite').orc(self.dirs.test_orc_path)

    @property
    def train_df(self):
        spark = SparkSession.builder.getOrCreate()
        return spark.read.schema(self.training_schema).orc(self.dirs.train_orc_path)

    @property
    def test_df(self):
        spark = SparkSession.builder.getOrCreate()
        return spark.read.schema(self.training_schema).orc(self.dirs.test_orc_path)

    # run

    def run_convert(self, overwrite: bool = True):
        if overwrite or not is_hdfs_exist(SparkSession.builder.enableHiveSupport().getOrCreate().sparkContext, self.dirs.converted_path):
            corpus_df = merge_orc_df(days=1, end_date_str=self.dirs.predict_date_str)
            apps_df = applist_multi_orc_df(days=1, end_date_str=self.dirs.predict_date_str)
            df = corpus_df.join(apps_df, 'userid', 'full')
            d = DataConverterTransformer()
            df = d.transform(df)
            df.write.mode('overwrite').orc(self.dirs.converted_path)

    def run_predict(self):
        spark = SparkSession.builder.enableHiveSupport().getOrCreate()
        with MmlShim():
            p = PipelineModel.load(self.dirs.model_path)
        df = spark.read.schema(self.converted_schema).orc(self.dirs.converted_path)
        df = p.transform(df).select('userid', 'prediction', 'prob')
        df.repartition(100).write.mode('overwrite').orc(self.dirs.result_path)

    def run_save(self, prob_threshold: float or None):
        spark = SparkSession.builder.enableHiveSupport().getOrCreate()
        df = spark.read.orc(self.dirs.result_path)
        if prob_threshold:
            df = df.where(f'prob>{prob_threshold}')
        save_userprofile(self.name, df)

    def run(self, predict_date_str, prob_threshold: float or None = None):
        assert predict_date_str is not None
        self.dirs.set_predict_date(predict_date_str)
        self.run_convert(overwrite=False)
        self.run_predict()
        self.run_save(prob_threshold=prob_threshold)

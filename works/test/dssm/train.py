import findspark
findspark.init()

import os
from pyspark.sql import SparkSession, Row
from pyspark.ml import Pipeline, PipelineModel
from autoflow.train import *
from torch import nn, optim
from model import DS
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from autoflow.train.evaluation import eval_binary


class FlatMapFunc:

    def __init__(self):
        self.converter = DataConverter()

    def map_func(self, row):
        res = row.asDict()
        res['lines'] = self.converter.convert_data(row.data)
        res['applist'] = self.converter.convert_applist(row.applist)
        res['apps'] = self.converter.convert_apps(row.applist)
        res['userprofile'] = self.converter.convert_userprofile(row.userprofile)
        return Row(**res)

    def flat_map_func(self, row):
        data = self.converter.convert_data(row.data)
        apps = self.converter.convert_apps(row.applist)
        applist = self.converter.convert_applist(row.applist)
        userprofile = self.converter.convert_userprofile(row.userprofile)

        def map_func_inner(x):
            arr = x.split(':')
            return {'vid': int(arr[0]),
                    'aid': int(arr[1]),
                    'label': float(arr[2]),
                    'lines': data,
                    'apps': apps,
                    'applist': applist,
                    'userprofile': userprofile}

        return map(map_func_inner, row.videos.split(','))

    def __call__(self, row):
        self.flat_map_func(row)


def train():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,6,7'

    spark = SparkSession.builder.master('local[*]').appName('test') \
        .config('spark.driver.memory', '20g') \
        .config('spark.driver.maxResultSize', '20g') \
        .config('spark.kryoserializer.buffer.max', '1g') \
        .config('spark.default.parallelism', 8) \
        .config('spark.sql.shuffle.partitions', 8) \
        .enableHiveSupport() \
        .getOrCreate()

    rdd = spark.sql('select * from dida.ks_label_weekfull').rdd\
        .sample(False, 0.0001).flatMap(FlatMapFunc().flat_map_func)
    train_df, test_df = spark.createDataFrame(rdd).randomSplit([0.1, 0.1], seed=1234)

    model = DS(userprofile_size=1384, applist_size=1000, vid_table_size=22261, aid_table_size=13727)
    classifier = PytorchLocalGPUClassifier(
        model=model,
        optimizer=optim.Adam(model.parameters(), lr=0.01),
        criterion=nn.MSELoss(),
        epochs=1,
        buffer_size=500,
        batch_size=100,
        input_cols=['vid', 'aid', 'lines', 'applist', 'userprofile'],
        label_col='label'
    )
    p = Pipeline(stages=[classifier]).fit(train_df)
    p.save('zxc/model/kuaishou/model')

    p = PipelineModel.load('zxc/model/kuaishou/model')

    # prediction
    prediction_df = p.transform(test_df).cache()

    # evaluator of spark
    auc = BinaryClassificationEvaluator(labelCol='label', rawPredictionCol='prob').evaluate(prediction_df)
    print(auc)

    # evaluate with picture
    eval_binary(prediction_df.collect(), label_col='label')


if __name__ == '__main__':
    train()

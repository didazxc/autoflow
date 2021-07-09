import findspark

findspark.init()

import os
import torch
from torch import nn, optim
from pyspark.sql import SparkSession
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml import PipelineModel
from autoflow.train import PytorchLocalGPUClassifier
from autoflow.train.evaluation import eval_binary
from autoflow.train.models.charCNN import CharCNN
from autoflow.train.estimators.stacking import StackingClassifier, StackingModel
from pyspark.ml.classification import LogisticRegression
from .run import trainProcess

train_orc_path = trainProcess.train_orc_path
test_orc_path = trainProcess.test_orc_path
model_path = trainProcess.model_path

schema = trainProcess.training_schema


def stacking_train():
    from mmlspark.lightgbm.LightGBMClassifier import LightGBMClassifier, LightGBMClassificationModel

    train_df = trainProcess.train_df.sample(False, 0.1, 17)  # .select('label', 'data')
    test_df = trainProcess.test_df.sample(False, 0.1, 17)  # .select('label', 'data')
    model = CharCNN([3, 5, 7])
    nlp_classifier = PytorchLocalGPUClassifier(
        model=model,
        optimizer=optim.Adam(model.parameters(), lr=0.01),
        criterion=nn.CrossEntropyLoss(),
        epochs=40,
        batch_size=50,
        input_cols=['data'],
        input_cols_type=[torch.long],
        label_col='label',
        label_col_type=torch.long)
    applist_classifier = LightGBMClassifier(
        objective='binary',
        learningRate=0.05,
        numIterations=100,
        numLeaves=31,
        featuresCol='applist',
        labelCol='label')
    s = StackingClassifier([nlp_classifier, applist_classifier], LogisticRegression(regParam=0.1, elasticNetParam=1.0))

    m: StackingModel = s.fit(train_df)
    PipelineModel([m]).save(model_path)

    prediction_df = m.transform(test_df).cache()
    prediction_df.printSchema()

    # evaluator of spark
    evaluator = BinaryClassificationEvaluator(labelCol='label', rawPredictionCol='rawPrediction')
    auc = evaluator.evaluate(prediction_df)
    print(auc)
    eval_binary(prediction_df.collect(), label_col='label')


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'

    spark = SparkSession.builder.master('local[*]') \
        .config("spark.submit.pyFiles",
                "/search/odin/opt2/spark/jars/mmlspark_2.11-0.18.1.jar") \
        .config('spark.yarn.dist.archives',
                'viewfs://marsX/user/dida/userprofile/lib/python3/Pytorch.zip#Python,' +
                'viewfs://marsX/user/dida/userprofile/lib/python3/hdfslib.zip#hdfslib') \
        .config('spark.pyspark.python', '~/anaconda3/envs/torch/bin/python') \
        .config('spark.driver.memory', '20g') \
        .config('spark.driver.maxResultSize', '20g') \
        .config('spark.kryoserializer.buffer.max', '1g') \
        .config('spark.default.parallelism', 8) \
        .config('spark.sql.shuffle.partitions', 8) \
        .getOrCreate()
    # .config("spark.jars.packages", "com.microsoft.ml.spark:mmlspark_2.11:0.18.1,com.microsoft.ml.lightgbm:lightgbmlib:2.2.350") \
    # .config("spark.jars",
    #                 "/root/.ivy2/jars/com.databricks_spark-xml_2.11-0.2.0.jar,"
    #                 "/root/.ivy2/jars/com.databricks_spark-xml_2.11-0.4.1.jar,"
    #                 "/root/.ivy2/jars/com.github.vowpalwabbit_vw-jni-8.7.0.2.jar,"
    #                 "/root/.ivy2/jars/com.jcraft_jsch-0.1.54.jar,"
    #                 "/root/.ivy2/jars/com.microsoft.cntk_cntk-2.4.jar,"
    #                 "/root/.ivy2/jars/com.microsoft.ml.lightgbm_lightgbmlib-2.2.350.jar,"
    #                 "/root/.ivy2/jars/com.microsoft.ml.spark_mmlspark_2.11-0.18.1.jar,"
    #                 "/root/.ivy2/jars/commons-codec_commons-codec-1.10.jar,"
    #                 "/root/.ivy2/jars/commons-logging_commons-logging-1.2.jar,"
    #                 "/root/.ivy2/jars/io.spray_spray-json_2.11-1.3.2.jar,"
    #                 "/root/.ivy2/jars/org.apache.httpcomponents_httpclient-4.5.6.jar,"
    #                 "/root/.ivy2/jars/org.apache.httpcomponents_httpcore-4.4.10.jar,"
    #                 "/root/.ivy2/jars/org.openpnp_opencv-3.2.0-1.jar,"
    #                 "/root/.ivy2/jars/org.scalactic_scalactic_2.11-3.0.5.jar,"
    #                 "/root/.ivy2/jars/org.scala-lang.modules_scala-xml_2.11-1.0.6.jar,"
    #                 "/root/.ivy2/jars/org.scala-lang_scala-reflect-2.11.12.jar,"
    #                 "/root/.ivy2/jars/org.scalatest_scalatest_2.11-3.0.5.jar") \

    trainProcess.train_convert()

    stacking_train()

#!/bin/sh

dir=$(cd "$(dirname "$0")";pwd)

time spark-submit \
  --master yarn \
  --deploy-mode cluster \
  --name zxc_gender_sample \
  --executor-cores 1 \
  --executor-memory 2g \
  --num-executors 100 \
  --driver-memory 2g \
  --driver-cores 1 \
  --conf spark.dynamicAllocation.enabled=true \
  --conf spark.dynamicAllocation.maxExecutors=1000 \
  --conf spark.driver.memoryOverhead=1g \
  --conf spark.executor.memoryOverhead=1g \
  --conf spark.driver.maxResultSize=2g \
  --conf spark.default.parallelism=2000  \
  --conf spark.sql.shuffle.partitions=2000  \
  --conf spark.task.cpus=1 \
  --conf spark.yarn.dist.archives=viewfs://marsX/user/dida/userprofile/lib/python3/Pytorch.zip#Python,viewfs://marsX/user/dida/userprofile/lib/python3/hdfslib.zip#hdfslib \
  --conf spark.pyspark.driver.python=./Python/bin/python3 \
  --conf spark.yarn.appMasterEnv.PYSPARK_PYTHON=./Python/bin/python3 \
  --conf spark.yarn.appMasterEnv.PYTHON_EGG_CACHE=./tmp/.cache/Python-Eggs \
  --conf spark.executorEnv.PYTHON_EGG_CACHE=./tmp/.cache/Python-Eggs \
  --conf spark.pyspark.python=./Python/bin/python3 \
  --py-files viewfs://marsX/user/dida/userprofile/lib/python3/autoflow.egg  \
  ${dir}/samples.py



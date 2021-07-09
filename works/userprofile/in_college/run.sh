#!/bin/sh

# hdfs dfs -put -f out.csv xxz/location.csv
dir=$(cd "$(dirname "$0")";pwd)

time spark-submit \
  --master yarn \
  --deploy-mode cluster \
  --name userprofile_in_college_test \
  --driver-cores 1 \
  --executor-cores 2 \
  --driver-memory 2g \
  --executor-memory 4g \
  --num-executors 100 \
  --conf spark.driver.memoryOverhead=1g \
  --conf spark.executor.memoryOverhead=2g \
  --conf spark.dynamicAllocation.enabled=true \
  --conf spark.dynamicAllocation.maxExecutors=750 \
  --conf spark.driver.maxResultSize=2g \
  --conf spark.default.parallelism=1000  \
  --conf spark.sql.shuffle.partitions=1000  \
  --conf spark.task.cpus=1 \
  --conf spark.yarn.dist.archives=viewfs://marsX/user/dida/userprofile/lib/python3/Pytorch.zip#Python,viewfs://marsX/user/dida/userprofile/lib/python3/hdfslib.zip#hdfslib \
  --conf spark.pyspark.driver.python=./Python/bin/python3 \
  --conf spark.pyspark.python=./Python/bin/python3 \
  --conf spark.yarn.appMasterEnv.PYSPARK_PYTHON=./Python/bin/python3 \
  --conf spark.yarn.appMasterEnv.PYTHON_EGG_CACHE=./tmp/.cache/Python-Eggs \
  --conf spark.executorEnv.PYTHON_EGG_CACHE=./tmp/.cache/Python-Eggs \
  --py-files viewfs://marsX/user/dida/userprofile/lib/python3/autoflow.egg  \
  ${dir}/aid_cnt.py

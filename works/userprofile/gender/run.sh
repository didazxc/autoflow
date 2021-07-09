#!/bin/sh

logdate=$1

dir=$(cd "$(dirname "$0")";pwd)

path1="/user/imeda/hive/warehouse/userinput.db/merge_orc/date=$logdate/000799_0"
path2="/user/imeda/hive/warehouse/userinput.db/merge_orc/date=$logdate/_temporary"

hadoop fs -test -e ${path1}
this1=$?
hadoop fs -test -e ${path2}
this2=$?
echo $this1
echo $this2

while  ! [ $this1 -eq 0 -a $this2 -eq 1 ]
do
    echo 'src file is not ready ,so wait for ten minutes'
    sleep 600
    hadoop fs -test -e ${path1}
    this1=$?
    hadoop fs -test -e ${path2}
    this2=$?
done


time spark-submit \
  --master yarn \
  --deploy-mode cluster \
  --name userprofile_gender_test \
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
  --conf spark.default.parallelism=2000  \
  --conf spark.sql.shuffle.partitions=2000  \
  --conf spark.task.cpus=1 \
  --conf spark.yarn.dist.archives=viewfs://marsX/user/dida/userprofile/lib/python3/Pytorch.zip#Python,viewfs://marsX/user/dida/userprofile/lib/python3/hdfslib.zip#hdfslib \
  --conf spark.pyspark.driver.python=./Python/bin/python3 \
  --conf spark.pyspark.python=./Python/bin/python3 \
  --conf spark.yarn.appMasterEnv.PYSPARK_PYTHON=./Python/bin/python3 \
  --conf spark.yarn.appMasterEnv.PYTHON_EGG_CACHE=./tmp/.cache/Python-Eggs \
  --conf spark.executorEnv.PYTHON_EGG_CACHE=./tmp/.cache/Python-Eggs \
  --packages com.microsoft.ml.spark:mmlspark_2.11:0.18.1 \
  --py-files viewfs://marsX/user/dida/userprofile/lib/python3/autoflow.egg  \
  ${dir}/run.py $logdate



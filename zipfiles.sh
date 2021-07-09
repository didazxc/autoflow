# dir=$(cd "$(dirname $0)" ; pwd)
# cd $dir/

# zip autoflow frame
# zip -r $dir/autoflow.zip autoflow
# hdfs dfs -put -f $dir/autoflow.zip /user/dida/userprofile/lib/python3/autoflow_dev.zip

# egg frame
python3 setup.py build_ext --inplace
python3 setup.py bdist_egg
# hdfs dfs -put -f dist/autoflow-*.egg /user/dida/userprofile/lib/python3/autoflow.egg

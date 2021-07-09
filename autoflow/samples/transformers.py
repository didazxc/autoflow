from pyspark.ml import Transformer
from pyspark import keyword_only, RDD, Row
from pyspark.sql import SparkSession
from pyspark.ml.param.shared import Param, Params, TypeConverters
from pyspark.ml.linalg import Vectors, _convert_to_vector, VectorUDT
from autoflow.samples.ruletree.rules import RuleTree
from pyspark.sql.types import StructType, StructField, StringType, ArrayType, IntegerType, MapType
import heapq
import scipy.sparse


def dense_to_sparse(vector):
    return _convert_to_vector(scipy.sparse.csc_matrix(vector.toArray()).T)


def list_to_sparse(vlist):
    return _convert_to_vector(scipy.sparse.csc_matrix(vlist).T)


class SamplesRuleExtractor(Transformer):
    """
    dataset must has a col named data
    """
    tag_rules = Param(Params._dummy(), "tag_rules", "Corresponding to tags one by one", typeConverter=TypeConverters.toListString)
    tags_col_name = Param(Params._dummy(), "tags_col_name", "tags_col_name", typeConverter=TypeConverters.toString)
    label_col_name = Param(Params._dummy(), "label_col_name", "label_col_name", typeConverter=TypeConverters.toString)
    data_col_name = Param(Params._dummy(), "data_col_name", "data_col_name", typeConverter=TypeConverters.toString)

    @keyword_only
    def __init__(self, tag_rules, tags_col_name='tags', label_col_name='label', data_col_name='data', apps_col_name='apps'):
        super().__init__()
        kwargs = self._input_kwargs
        self._setDefault(tags_col_name='tags', label_col_name='label', data_col_name='data')
        self._set(**kwargs)

    def _transform(self, dataset):
        spark: SparkSession = dataset.sql_ctx
        rdd: RDD = dataset.rdd
        sc = rdd.context
        tag_rules_bc = sc.broadcast(self.getOrDefault(self.tag_rules))
        tags_col_name_bc = sc.broadcast(self.getOrDefault(self.tags_col_name))
        label_col_name_bc = sc.broadcast(self.getOrDefault(self.label_col_name))
        data_col_name_bc = sc.broadcast(self.getOrDefault(self.data_col_name))
        apps_col_name_bc = sc.broadcast(self.getOrDefault(self.apps_col_name))

        def map_partitions_func(iterator):
            tags_col_name: str = tags_col_name_bc.value
            label_col_name: str = label_col_name_bc.value
            data_col_name: str = data_col_name_bc.value
            apps_col_name: str = apps_col_name_bc.value
            s: RuleTree = RuleTree(tag_rules_bc.value)
            for row in iterator:
                s.reset()
                d = row.asDict() if isinstance(row, Row) else row
                if data_col_name in d:
                    s.calc_scores(b'data', d[data_col_name])
                if apps_col_name in d:
                    s.calc_scores(b'apps', d[apps_col_name])
                d[tags_col_name] = list(s.get_score())
                top2 = heapq.nlargest(2, d[tags_col_name])
                if top2[0] > top2[1]:
                    d[label_col_name] = float(d[tags_col_name].index(top2[0]))
                    yield d
        rdd = rdd.mapPartitions(map_partitions_func)
        return dataset if rdd.isEmpty() else spark.createDataFrame(rdd.mapPartitions(map_partitions_func))


class TagsRuleExtractor(Transformer):
    """
    dataset must has a col named data
    """
    tag_rules = Param(Params._dummy(), "tag_rules", "Corresponding to tags one by one", typeConverter=TypeConverters.toListString)
    tags_col_name = Param(Params._dummy(), "tags_col_name", "tags_col_name", typeConverter=TypeConverters.toString)
    data_col_name = Param(Params._dummy(), "data_col_name", "data_col_name", typeConverter=TypeConverters.toString)
    scores_col_name = Param(Params._dummy(), "scores_col_name", "data_col_name", typeConverter=TypeConverters.toString)

    @keyword_only
    def __init__(self, tag_rules, tags_col_name='tags', data_col_name='data', apps_col_name='apps', scores_col_name='scores'):
        super().__init__()
        kwargs = self._input_kwargs
        self._setDefault(tags_col_name='tags', data_col_name='data')
        self._set(**kwargs)

    def _transform(self, dataset):
        spark: SparkSession = dataset.sql_ctx
        rdd: RDD = dataset.rdd
        sc = rdd.context
        tag_rules_bc = sc.broadcast(self.getOrDefault(self.tag_rules))
        tags_col_name_bc = sc.broadcast(self.getOrDefault(self.tags_col_name))
        data_col_name_bc = sc.broadcast(self.getOrDefault(self.data_col_name))
        apps_col_name_bc = sc.broadcast(self.getOrDefault(self.apps_col_name))
        scores_col_name_bc = sc.broadcast(self.getOrDefault(self.scores_col_name))

        def map_partitions_func(iterator):
            tags_col_name: str = tags_col_name_bc.value
            data_col_name: str = data_col_name_bc.value
            apps_col_name: str = apps_col_name_bc.value
            scores_col_name: str = scores_col_name_bc.value
            s: RuleTree = RuleTree(tag_rules_bc.value)
            for row in iterator:
                s.reset()
                d = row.asDict() if isinstance(row, Row) else row
                if data_col_name in d:
                    s.calc_scores(b'data', d[data_col_name])
                if apps_col_name in d:
                    s.calc_scores(b'apps', d[apps_col_name])
                d[tags_col_name] = s.get_score()
                d[scores_col_name] = list(s.get_sub_scores())
                yield d
        rdd = rdd.mapPartitions(map_partitions_func)
        return dataset if rdd.isEmpty() else spark.createDataFrame(rdd.mapPartitions(map_partitions_func))


class TagsUserExtractor(Transformer):
    """
    dataset must has a col named userid
    data or apps at least has one
    calc_tag_directly func need date col
    """
    tag_rules = Param(Params._dummy(), "tag_rules", "Corresponding to tags one by one", typeConverter=TypeConverters.toListString)
    userid_col_name = Param(Params._dummy(), "userid_col_name", "userid_col_name", typeConverter=TypeConverters.toString)
    date_col_name = Param(Params._dummy(), "date_col_name", "date_col_name", typeConverter=TypeConverters.toString)
    data_col_name = Param(Params._dummy(), "data_col_name", "data_col_name", typeConverter=TypeConverters.toString)
    apps_col_name = Param(Params._dummy(), "apps_col_name", "apps_col_name", typeConverter=TypeConverters.toString)

    @keyword_only
    def __init__(self, tag_rules, userid_col_name='userid', data_col_name='data', apps_col_name='apps', date_col_name='date'):
        super().__init__()
        kwargs = self._input_kwargs
        self._setDefault(userid_col_name='userid', data_col_name='data', apps_col_name='apps', date_col_name='date')
        self._set(**kwargs)

    def _transform(self, dataset):
        spark: SparkSession = dataset.sql_ctx
        rdd: RDD = dataset.rdd
        sc = rdd.context
        userid_col_name = self.getOrDefault(self.userid_col_name)
        userid_col_name_bc = sc.broadcast(userid_col_name)
        date_col_name = self.getOrDefault(self.date_col_name)
        date_col_name_bc = sc.broadcast(date_col_name)
        tag_rules_bc = sc.broadcast(self.getOrDefault(self.tag_rules))
        data_col_name_bc = sc.broadcast(self.getOrDefault(self.data_col_name))
        apps_col_name_bc = sc.broadcast(self.getOrDefault(self.apps_col_name))

        def map_partitions_func(iterator):
            userid_col_name: str = userid_col_name_bc.value
            date_col_name: str = date_col_name_bc.value
            data_col_name: str = data_col_name_bc.value
            apps_col_name: str = apps_col_name_bc.value
            s: RuleTree = RuleTree(tag_rules_bc.value)
            for row in iterator:
                s.reset()
                d = row.asDict() if isinstance(row, Row) else row
                if data_col_name in d:
                    s.calc_scores(b'data', d[data_col_name])
                if apps_col_name in d:
                    s.calc_scores(b'apps', d[apps_col_name])
                scores = list(s.get_score())
                if any(scores):
                    yield Row(d[userid_col_name], scores, d[date_col_name])

        schema = StructType([StructField(userid_col_name, StringType(), True),
                             StructField('tags', ArrayType(IntegerType()), False),
                             StructField(date_col_name, StringType(), False)])
        return spark.createDataFrame(rdd.mapPartitions(map_partitions_func), schema)

    def calc_tag_directly(self, dataset, tags):
        spark: SparkSession = dataset.sql_ctx
        rdd: RDD = dataset.rdd
        sc = rdd.context
        userid_col_name = self.getOrDefault(self.userid_col_name)
        userid_col_name_bc = sc.broadcast(userid_col_name)
        date_col_name = self.getOrDefault(self.date_col_name)
        date_col_name_bc = sc.broadcast(date_col_name)
        tag_rules_bc = sc.broadcast(self.getOrDefault(self.tag_rules))
        data_col_name_bc = sc.broadcast(self.getOrDefault(self.data_col_name))
        apps_col_name_bc = sc.broadcast(self.getOrDefault(self.apps_col_name))
        tags_bc = sc.broadcast(tags)

        def map_partitions_func(iterator):
            userid_col_name: str = userid_col_name_bc.value
            date_col_name: str = date_col_name_bc.value
            data_col_name: str = data_col_name_bc.value
            apps_col_name: str = apps_col_name_bc.value
            s: RuleTree = RuleTree(tag_rules_bc.value)
            for row in iterator:
                s.reset()
                d = row.asDict() if isinstance(row, Row) else row
                if data_col_name in d:
                    s.calc_scores(b'data', d[data_col_name])
                if apps_col_name in d:
                    s.calc_scores(b'apps', d[apps_col_name])
                scores = list(s.get_score())
                if any(scores):
                    user_tags = dict(filter(lambda x: x[1] > 0, zip(tags_bc.value, scores)))
                    yield Row(d[userid_col_name], user_tags, d[date_col_name])

        schema = StructType([StructField(userid_col_name, StringType(), True),
                             StructField('tags', MapType(StringType(), IntegerType()), False),
                             StructField(date_col_name, StringType(), False)])
        return spark.createDataFrame(rdd.mapPartitions(map_partitions_func), schema)

    def calc_sub_scores(self, dataset):
        spark: SparkSession = dataset.sql_ctx
        rdd: RDD = dataset.rdd
        sc = rdd.context
        userid_col_name = self.getOrDefault(self.userid_col_name)
        userid_col_name_bc = sc.broadcast(userid_col_name)
        tag_rules_bc = sc.broadcast(self.getOrDefault(self.tag_rules))
        data_col_name_bc = sc.broadcast(self.getOrDefault(self.data_col_name))
        apps_col_name_bc = sc.broadcast(self.getOrDefault(self.apps_col_name))

        def map_partitions_func(iterator):
            userid_col_name: str = userid_col_name_bc.value
            data_col_name: str = data_col_name_bc.value
            apps_col_name: str = apps_col_name_bc.value
            s: RuleTree = RuleTree(tag_rules_bc.value)
            for row in iterator:
                s.reset()
                d = row.asDict() if isinstance(row, Row) else row
                if data_col_name in d:
                    s.calc_scores(b'data', d[data_col_name])
                if apps_col_name in d:
                    s.calc_scores(b'apps', d[apps_col_name])
                sub_scores = list_to_sparse(list(s.get_sub_scores()))
                if sub_scores.numNonzeros():
                    yield Row(d[userid_col_name], sub_scores)

        schema = StructType([StructField(userid_col_name, StringType(), True),
                             StructField('sub_scores', VectorUDT(), False)])
        return spark.createDataFrame(rdd.mapPartitions(map_partitions_func), schema)

    def calc_tag(self, dataset):
        spark: SparkSession = dataset.sql_ctx
        rdd: RDD = dataset.rdd
        tag_rules_bc = rdd.context.broadcast(self.getOrDefault(self.tag_rules))
        userid_col_name = self.getOrDefault(self.userid_col_name)
        userid_col_name_bc = rdd.context.broadcast(userid_col_name)

        def reduce_fn(scores_a, scores_b):
            return dense_to_sparse(Vectors.dense(scores_a.toArray())+Vectors.dense(scores_b.toArray()))

        def map_partitions_func(iterator):
            s: RuleTree = RuleTree(tag_rules_bc.value)
            for row in iterator:
                s.set_sub_scores(list(map(lambda x: int(x), row[1].toArray())))
                tags = list(s.get_score())
                if any(tags):
                    yield Row(row[0], row[1], tags)

        if not rdd.isEmpty():
            rdd = rdd.map(lambda row: (row[userid_col_name_bc.value], row['sub_scores']))\
                .reduceByKey(reduce_fn).mapPartitions(map_partitions_func)

        schema = StructType([StructField(userid_col_name, StringType(), True),
                             StructField('sub_scores', VectorUDT(), False),
                             StructField('tags', ArrayType(IntegerType()), False)])
        return spark.createDataFrame(rdd, schema)

    def convert_tags(self, dataset, tags):
        spark: SparkSession = dataset.sql_ctx
        rdd: RDD = dataset.rdd
        userid_col_name = self.getOrDefault(self.userid_col_name)
        userid_col_name_bc = rdd.context.broadcast(userid_col_name)
        tags_bc = rdd.context.broadcast(tags)

        def map_fn(row):
            user_tags = dict(filter(lambda x: x[1] > 0, zip(tags_bc.value, row['tags'])))
            return Row(row[userid_col_name_bc.value], user_tags)

        schema = StructType([StructField(userid_col_name, StringType(), True),
                             StructField('tags', MapType(StringType(), IntegerType()), False)])
        return spark.createDataFrame(rdd.map(map_fn), schema)

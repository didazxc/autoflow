from pyspark.sql import DataFrame
from pyspark.sql.functions import udf
from pyspark.sql.types import DoubleType
from pyspark.ml import Estimator, Model, PipelineModel, Pipeline
from pyspark.ml.util import MLReadable, MLWritable, DefaultParamsWriter, DefaultParamsReader
from pyspark.ml.param.shared import HasPredictionCol, HasRawPredictionCol, HasProbabilityCol, HasFeaturesCol
from typing import List
from pyspark.ml.feature import VectorAssembler


udf_probability_max = udf(lambda v: float(max(v.values)), DoubleType())


class StackingModel(Model, MLWritable, MLReadable):

    def __init__(self, models: PipelineModel=None, meta_model: PipelineModel=None):
        super().__init__()
        self.models: PipelineModel = models
        self.meta_model: PipelineModel = meta_model

    def _transform(self, dataset: DataFrame):
        dataset = self.models.transform(dataset)
        dataset = self.meta_model.transform(dataset)
        col_names = dataset.schema.names
        if 'prob' not in col_names and 'probability' in col_names:
            dataset = dataset.withColumn('prob', udf_probability_max('probability'))
        return dataset

    def write(self):
        return self

    def save(self, path):
        DefaultParamsWriter(self).save(path)
        self.models.save(f'{path}/models')
        self.meta_model.save(f'{path}/meta_model')

    @classmethod
    def read(cls):
        return cls

    @classmethod
    def load(cls, path):
        m: StackingModel = DefaultParamsReader(cls).load(path)
        m.models = PipelineModel.load(f'{path}/models')
        m.meta_model = PipelineModel.load(f'{path}/meta_model')
        return m


class StackingClassifier(Estimator):

    def __init__(self, classifiers: List[Estimator], meta_classifier: Estimator and HasFeaturesCol):
        super().__init__()
        self.classifiers = classifiers
        self.meta_classifier = meta_classifier

    @staticmethod
    def change_col_name(m: Model or Estimator, i: int):
        if isinstance(m, HasRawPredictionCol) or hasattr(m, 'rawPredictionCol'):
            m.set(m.rawPredictionCol, f'_rawPred_{i}')
        if isinstance(m, HasPredictionCol) or hasattr(m, 'predictionCol'):
            m.set(m.predictionCol, f'_pred_{i}')
        if isinstance(m, HasProbabilityCol) or hasattr(m, 'probabilityCol'):
            m.set(m.probabilityCol, f'_prob_{i}')

    def _fit(self, dataset: DataFrame):
        models = []
        for i, classifier in enumerate(self.classifiers):
            self.change_col_name(classifier, i)
            m = classifier.fit(dataset)
            self.change_col_name(m, i)
            models.append(m)
        cols = []
        for i, m in enumerate(models):
            dataset: DataFrame = m.transform(dataset)
            cols.append(f'_prob_{i}')
            dataset = dataset.drop(f'_rawPred_{i}', f'_pred_{i}')
        assembler = VectorAssembler(inputCols=cols, outputCol="_stacking_features")
        self.meta_classifier.set(self.meta_classifier.featuresCol, '_stacking_features')
        meta_model = Pipeline(stages=[assembler, self.meta_classifier]).fit(dataset)
        return StackingModel(PipelineModel(models), meta_model)

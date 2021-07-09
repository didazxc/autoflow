from pyspark.ml.util import DefaultParamsReader
from unittest import mock


class MmlShim(object):
    """
    When pipelineModel load mmlspark, there will be a exception:
        AttributeError: module 'com.microsoft.ml.spark.lightgbm' has no attribute 'LightGBMClassificationModel'
    because 'com.microsoft.ml.spark' should be 'mmlspark'.
    So we use this shim to fix it.
    Useage:
        with MmlShim():
            PipelineModel.load(model_path)
    """
    mangled_name = '_DefaultParamsReader__get_class'
    prev_get_clazz = getattr(DefaultParamsReader, mangled_name)

    @classmethod
    def __get_class(cls, clazz):
        try:
            return cls.prev_get_clazz(clazz)
        except AttributeError as outer:
            try:
                alt_clazz = clazz.replace('com.microsoft.ml.spark', 'mmlspark')
                return cls.prev_get_clazz(alt_clazz)
            except AttributeError:
                raise outer

    def __enter__(self):
        self.mock = mock.patch.object(DefaultParamsReader, self.mangled_name, self.__get_class)
        self.mock.__enter__()
        return self

    def __exit__(self, *exc_info):
        self.mock.__exit__(*exc_info)
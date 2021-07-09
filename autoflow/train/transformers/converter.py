import os
import json
from collections import OrderedDict
from pyspark.ml import Transformer
from pyspark import keyword_only, RDD, SparkContext, SparkFiles
from pyspark.sql import DataFrame, SparkSession
from pyspark.ml.util import DefaultParamsWritable, DefaultParamsReadable
from pyspark.ml.param.shared import *
from pyspark.ml.linalg import SparseVector, DenseVector


class DataConverter:

    def __init__(self, max_length=1000, need_app=True, need_query=True, app_length=None, vocab_file=None, app_file=None,
                 userprofile_file=None, need_pad_zero=True):
        self.max_length = max_length
        self.need_app = need_app
        self.need_query = need_query
        self.need_pad_zero = need_pad_zero

        # chars_dict
        parent_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        if vocab_file is None:
            vocab_file = os.path.join(parent_dir, 'conf/vocab.txt')
        with open(vocab_file, 'r', encoding='utf8') as f:
            self.chars_dict = {char.strip(): index for index, char in enumerate(f)}
        self.vocab_chars_size = len(self.chars_dict)

        # apps_dict, combine to lines_dict
        if app_file is None:
            app_file = os.path.join(parent_dir, 'conf/applist.txt')
        with open(app_file, 'r', encoding='utf8') as f:
            self.apps_dict = {app.strip(): index for index, app in enumerate(f)}
            self.chars_dict.update({app.strip(): self.vocab_chars_size + index for index, app in enumerate(f)})
        self.app_length = app_length if app_length else len(self.apps_dict)

        # size of vocab
        self.vocab_size = len(self.chars_dict)

        # userprofile_dict:OrderedDict
        if userprofile_file is None:
            userprofile_file = os.path.join(parent_dir, 'conf/userprofile.json')
        with open(userprofile_file, 'r', encoding='utf8') as u_file:
            self.u_dict: OrderedDict = json.load(u_file, object_pairs_hook=OrderedDict)

        # sizes of every userprofile tag
        self.u_dict_sizes: OrderedDict = OrderedDict(
            {k: len(self.u_dict[k]) for k in self.u_dict if k != 'device' and k != 'dpi'})

        # total size of userprofile tags
        self.userprofile_size = sum(i for i in self.u_dict_sizes.values())

    # convert data to index in chars_dict
    def convert_data(self, iter_data):
        """
        data -- app:timestamp:corpus[:is_query]
        from userinput.merge_orc
        """
        if iter_data is None:
            return [0] * self.max_length
        res = []
        length = 0
        for item in iter_data:
            arr = item.split(':', 3)
            words = []
            if self.need_app:
                app = self.chars_dict.get(arr[0], 1)
                words.append(app)
            if self.need_query:
                query = 4 if len(arr) == 4 and arr[3] == '1' else 5
                words.append(query)
            words += [self.chars_dict.get(w, 1) for w in arr[2]]
            words.append(2)
            words_len = len(words)
            rest_length = self.max_length - length
            if rest_length > words_len:
                res += words
                length += words_len
            else:
                res += words[:rest_length]
                length = self.max_length
                break
        if self.need_pad_zero and length < self.max_length:
            res += [0] * (self.max_length - length)
        return res

    def convert_data_vector(self, iter_data):
        return DenseVector(self.convert_data(iter_data))

    # convert userprofile to list (one-hot tags)
    def convert_userprofile(self, userprofile_dict: dict):
        res = [0.0] * self.userprofile_size
        if userprofile_dict is not None:
            n = 0
            for k, s in self.u_dict_sizes.items():
                if k in userprofile_dict and userprofile_dict[k] is not None:
                    i = self.u_dict[k][userprofile_dict[k]]
                    res[i + n] = 1.0
                n += s
        return res

    # convert app to list

    def convert_applist(self, iter_applist):
        """
        applist -- app:freq:dur:is_sys
        from applist.applist_multi_orc
        """
        res = [0.0] * self.app_length
        if iter_applist is None:
            return res
        for w in iter_applist:
            try:
                index = self.apps_dict[w.split(':', 1)[0]]
            except KeyError:
                pass
            except Exception as e:
                print(e)
            else:
                res[index] = 1.0
        return res

    def convert_app_multi(self, iter_applist):
        """
        applist -- app:freq:dur:is_sys
        from applist.applist_multi_orc
        combine list,freq,dur,isSys
        """
        vec_list = [0.0] * self.app_length
        vec_freq = [0.0] * self.app_length
        vec_dur = [0.0] * self.app_length
        vec_sys = [0.0] * self.app_length
        if iter_applist is not None:
            for w in iter_applist:
                try:
                    app, is_sys, freq, dur, _ = w.split(':', 4)
                    index = self.apps_dict[app]
                    vec_list[index] = 1.0
                    vec_freq[index] = float(freq) if freq != '' else 0.0
                    vec_dur[index] = float(dur) if dur != '' else 0.0
                    vec_sys[index] = float(is_sys) if is_sys != '2' and is_sys != '' else -1.0
                except Exception as e:
                    pass
                    # print(w, e)
        return vec_list, vec_freq, vec_dur, vec_sys

    def convert_apps(self, iter_applist):
            """ convert applist to an index list, every app will be converted to index in chars_dict """
            if iter_applist is None:
                return [0] * self.app_length
            res = [self.chars_dict.get(w.split(':', 1)[0], 1) for i, w in enumerate(iter_applist) if
                   i < self.app_length]
            res_len = len(res)
            if self.need_pad_zero and res_len < self.app_length:
                res += [0] * (self.app_length - res_len)
            return res

    # convert app to SparseVector

    def _convert_app_multi_dict(self, iter_applist):
        """
        applist -- app:freq:dur:is_sys
        from applist.applist_multi_orc
        combine list,freq,dur,isSys
        """
        vec_list = dict()
        vec_freq = dict()
        vec_dur = dict()
        vec_sys = dict()
        if iter_applist is not None:
            for w in iter_applist:
                try:
                    app, is_sys, freq, dur, _ = w.split(':', 4)
                    index = self.apps_dict[app]
                    vec_list[index] = 1.0
                    if freq != '':
                        vec_freq[index] = float(freq)
                    if dur != '':
                        vec_dur[index] = float(dur)
                    vec_sys[index] = float(is_sys) if is_sys != '2' and is_sys != '' else -1.0
                except Exception as e:
                    pass
        return vec_list, vec_freq, vec_dur, vec_sys

    def convert_app_multi_vector(self, iter_applist) -> (SparseVector, SparseVector, SparseVector, SparseVector):
        vec_list, vec_freq, vec_dur, vec_sys = self._convert_app_multi_dict(iter_applist)
        return SparseVector(self.app_length, vec_list), SparseVector(self.app_length, vec_freq), \
               SparseVector(self.app_length, vec_dur), SparseVector(self.app_length, vec_sys)

    def convert_app_multi_vector_with_data(self, iter_applist, iter_data):
        vec_list, vec_freq, vec_dur, vec_sys = self._convert_app_multi_dict(iter_applist)
        if iter_data is not None:
            for item in iter_data:
                app = item.split(':', 1)[0]
                try:
                    index = self.apps_dict[app]
                    vec_list[index] = 1.0
                except Exception as e:
                    pass
        return SparseVector(self.app_length, vec_list), SparseVector(self.app_length, vec_freq), \
               SparseVector(self.app_length, vec_dur), SparseVector(self.app_length, vec_sys)


class DataConverterTransformer(Transformer, DefaultParamsWritable, DefaultParamsReadable):
    max_length = Param(Params._dummy(), "max_length", "max length", typeConverter=TypeConverters.toInt)
    need_app = Param(Params._dummy(), "need_app", "need app", typeConverter=TypeConverters.toBoolean)
    need_query = Param(Params._dummy(), "need_query", "need_query", typeConverter=TypeConverters.toBoolean)
    app_length = Param(Params._dummy(), "app_length", "app_length", typeConverter=TypeConverters.toInt)
    vocab_file = Param(Params._dummy(), "vocab_file", "vocab_file", typeConverter=TypeConverters.toString)
    app_file = Param(Params._dummy(), "app_file", "app_file", typeConverter=TypeConverters.toString)
    userprofile_file = Param(Params._dummy(), "userprofile_file", "file path", typeConverter=TypeConverters.toString)
    need_pad_zero = Param(Params._dummy(), "need_pad_zero", "need pad zero", typeConverter=TypeConverters.toBoolean)

    @keyword_only
    def __init__(self, max_length=1000, need_app=True, need_query=True, app_length=None, vocab_file=None, app_file=None,
                 userprofile_file=None, need_pad_zero=True):
        super().__init__()
        kwargs = self._input_kwargs
        self._set(**kwargs)
        self._setDefault(vocab_file='viewfs:///user/dida/zxc/lib/autoflow/vocab.txt',
                         app_file='viewfs:///user/dida/zxc/lib/autoflow/applist.txt',
                         userprofile_file='viewfs:///user/dida/zxc/lib/autoflow/userprofile.json')
        self.data_converter = None

    def setNeedPadZero(self, value):
        return self._set(need_pad_zero=value)

    def create_data_converter(self, sc: SparkContext):
        if self.data_converter is None:
            f = {k: self.getOrDefault(k)
                 for k in [self.vocab_file, self.app_file, self.userprofile_file]}
            for i in f.values():
                sc.addFile(i)
            extra = {k: SparkFiles.get(os.path.basename(v)) for k, v in f.items()}
            kwargs = self.extractParamMap(extra)
            self.data_converter = DataConverter(**{k.name: v for k, v in kwargs.items()})

    def _transform(self, dataset: DataFrame):
        spark: SparkSession = dataset.sql_ctx
        rdd: RDD = dataset.rdd
        sc = rdd.context
        self.create_data_converter(sc)
        data_converter_bc = sc.broadcast(self.data_converter)

        def map_partitions_func(iterator):
            data_converter: DataConverter = data_converter_bc.value
            for row in iterator:
                d = row.asDict()
                if 'data' in d:
                    data = d['data']
                    d['data'] = data_converter.convert_data(data)
                    d['applist'], d['appfreq'], d['appdur'], d[
                        'appsys'] = data_converter.convert_app_multi_vector_with_data(
                        iter_applist=d['applist'] if 'applist' in d else None, iter_data=data)
                elif 'applist' in d:
                    d['applist'], d['appfreq'], d['appdur'], d['appsys'] = data_converter.convert_app_multi_vector(
                        d['applist'])
                yield d

        return spark.createDataFrame(rdd.mapPartitions(map_partitions_func))

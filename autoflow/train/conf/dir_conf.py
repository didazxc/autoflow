
import os


class DirConf:
    root = 'userprofile'
    lib = os.path.join(root, 'lib')
    prepare = os.path.join(root, 'prepare')  # prepare data for models
    model = os.path.join(root, 'model')
    stat = os.path.join(root, 'stat')  # statistics


class ModelDirConf(DirConf):

    def __init__(self, name, version, model_name, sample_date_str, predict_date_str):
        self.version = version

        public_dir = os.path.join(self.root, 'prepare', version)
        self.public_dir = public_dir
        self.convert_dir = os.path.join(public_dir, 'convert')

        directory = os.path.join(self.root, 'model', name, version)
        self.directory = directory
        self.sample_dir = os.path.join(directory, 'sample')
        self.result_dir = os.path.join(directory, 'result')
        self.model_dir = os.path.join(directory, 'model')

        self.model_path = os.path.join(self.model_dir, model_name)

        self._set_sample_date(sample_date_str)
        self._set_predict_date(predict_date_str)

    def _set_sample_date(self, date_str):
        sample_dir = self.sample_dir
        self.sample_date_str = date_str
        self.sample_corpus_orc_path = f'{sample_dir}/corpus_{date_str}.orc'
        self.sample_corpus_balance_orc_path = f'{sample_dir}/corpus_balance_{date_str}.orc'
        self.sample_applist_orc_path = f'{sample_dir}/applist_{date_str}.orc'
        self.sample_applist_balance_orc_path = f'{sample_dir}/applist_balance_{date_str}.orc'
        self.train_orc_path = f'{sample_dir}/train_{date_str}.orc'
        self.test_orc_path = f'{sample_dir}/test_{date_str}.orc'

    def _set_predict_date(self, date_str):
        self.predict_date_str = date_str
        self.converted_path = f'{self.convert_dir}/{date_str}.orc'
        self.result_path = f'{self.result_dir}/{date_str}.orc'

    def set_sample_date(self, date_str):
        self._set_sample_date(date_str)

    def set_predict_date(self, date_str):
        self._set_predict_date(date_str)

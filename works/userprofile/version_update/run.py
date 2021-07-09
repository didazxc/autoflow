import sys
from autoflow.train.process import TrainProcess

# use for train, should set_date_str in prediction task
trainProcess = TrainProcess(name='version_update', sample_date_str='20210510', version='v1.0.0',
                            model_name='train_stacking', only_public_version=True)

if __name__ == '__main__':
    date = sys.argv[1]
    trainProcess.run(predict_date_str=date)

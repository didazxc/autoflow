import io
import os
import time
from typing import List
from pyspark import keyword_only, RDD
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import udf
from pyspark.ml import Estimator, Model
from pyspark.ml.param.shared import *
from pyspark.ml.util import MLReadable, MLWritable, DefaultParamsWriter, DefaultParamsReader
from pyspark.sql.types import ArrayType, DoubleType, StructField
from pyspark.ml.linalg import DenseVector
import torch
from torch.utils.data import DataLoader
import torch.distributed as dist
from autoflow.train.data import IterDataset, FlatMapShuffleDataset
from pyspark.ml.linalg import VectorUDT


udf_vec2arr = udf(lambda vector: vector.toArray().tolist(), ArrayType(DoubleType()))


class Config:
    def __init__(self, model, optimizer=None, criterion=None, buffer_size=500, batch_size=50, epochs=1,
                 input_cols=None, input_cols_type=None,
                 label_col='label', label_col_type=None,
                 checkpoint_path='./model.checkpoint', print_every=100,
                 early_stop=10000):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion

        # data_loader_fn(iterator) =
        # DataLoader(FlatMapShuffleDataset(iterator,config.flat_map_fn,config.buffer_size,config.epochs),config.batch_size)
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.epochs = epochs

        # outputs = model(**{name: rows[name].to(device) for name in config.input_cols})
        # loss = criterion(*outputs, rows[label_col].to(device))
        self.input_cols = input_cols if input_cols else ['data']
        self.label_col = label_col
        self.input_cols_type = input_cols_type
        self.label_col_type = label_col_type

        # save and print
        self.checkpoint_path = checkpoint_path
        self.print_every = print_every
        self.early_stop = early_stop


class LocalGPUDistTrain:
    @staticmethod
    def average_gradients(model):
        size = float(dist.get_world_size())
        for param in model.parameters():
            dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
            param.grad.data /= size

    @staticmethod
    def process(rank, data_loader, conf: Config):

        optimizer = conf.optimizer
        criterion = conf.criterion
        input_cols = conf.input_cols
        label_col = conf.label_col
        checkpoint_path = conf.checkpoint_path
        print_every = conf.print_every
        early_stop = conf.early_stop

        gpu_num = torch.cuda.device_count()
        device_rank = rank % gpu_num
        device = torch.device(f"cuda:{device_rank}")
        world_size = float(dist.get_world_size())

        model = conf.model.to(device)
        # load model
        if os.path.exists(checkpoint_path):
            model.load_state_dict(torch.load(checkpoint_path, map_location={'cuda:0': f"cuda:{device_rank}"}))
            print(f'Rank {rank} have load the checkpoint model from {os.path.abspath(checkpoint_path)}')
        dist.barrier()

        # change model type to train
        model.train()
        stop_i = 0
        min_loss = float('inf')
        time_start = time.time()
        all_losses = []
        for index_batches, rows in enumerate(data_loader):
            # train
            optimizer.zero_grad()
            outputs = model(**{name: rows[name].to(device) for name in input_cols})
            # print(type(outputs))
            if isinstance(outputs, tuple):
                loss = criterion(*outputs, rows[label_col].to(device))
            else:
                loss = criterion(outputs, rows[label_col].to(device))
            loss.backward()
            try:
                LocalGPUDistTrain.average_gradients(model)
            except Exception as e:
                print(e)
                break
            optimizer.step()
            # save model and early stop
            dist.reduce(loss, dst=0, op=dist.ReduceOp.SUM)
            if rank == 0:
                cur_loss = loss.item() / world_size
                all_losses.append(cur_loss)
                if index_batches % print_every == 0:
                    print(f'index {index_batches} min_loss:{min_loss} cur_loss:{cur_loss} stop_i:{stop_i}')
                if cur_loss < min_loss:
                    min_loss = cur_loss
                    torch.save(model.state_dict(), checkpoint_path)
                    print(f'checkpoint have been saved. index_batches:{index_batches}, min_loss:{min_loss}')
                    stop_i = 0
                else:
                    stop_i += 1
                if stop_i > early_stop:
                    print(f'early_stop:{stop_i}')
                    break
        # print usage time
        print(f'Rank {rank}, time:{time.time() - time_start}s')
        # plot loss
        if rank == 0:
            import matplotlib.pyplot as plt
            plt.figure()
            plt.plot(all_losses)
            plt.show()
            return model.state_dict()
        else:
            return None

    @staticmethod
    def train(rdd: RDD, conf: Config, world_size=0):
        """
        input_cols must be array, like list, np.array
        """
        world_size = world_size if world_size else torch.cuda.device_count()
        config_bc = rdd.context.broadcast(conf)

        def train_map_partitions_with_index(rank, iterator):
            config = config_bc.value

            def flat_map_fn(row):
                if config.input_cols_type:
                    res_dict = {name: torch.tensor(row[name], dtype=ty) for name, ty
                                in zip(config.input_cols, config.input_cols_type)}
                else:
                    res_dict = {name: torch.tensor(row[name]) for name in config.input_cols}
                if config.label_col_type:
                    res_dict[config.label_col] = torch.tensor(row[config.label_col], dtype=config.label_col_type)
                else:
                    res_dict[config.label_col] = torch.tensor(row[config.label_col])
                return [res_dict]

            data_loader = DataLoader(
                FlatMapShuffleDataset(iterator, flat_map_fn, buffer_size=config.buffer_size,
                                      tee_n=config.epochs), batch_size=config.batch_size)

            os.environ['MASTER_ADDR'] = '127.0.0.1'
            os.environ['MASTER_PORT'] = '29500'
            dist.init_process_group(backend='gloo', rank=rank, world_size=world_size)
            state_dict = LocalGPUDistTrain.process(rank, data_loader, config)
            dist.destroy_process_group()
            return [state_dict] if state_dict else []

        arr = rdd.repartition(world_size).mapPartitionsWithIndex(train_map_partitions_with_index).collect()
        if len(arr):
            conf.model.load_state_dict(arr[0])

    @staticmethod
    def prediction(rdd: RDD, conf: Config, rawPredictionCol:str = 'rawPrediction', predictionCol:str = 'prediction', probabilityCol:str = 'probability') -> RDD:
        """
        input_cols must be array, like list, np.array
        """
        sc = rdd.context
        model_bc = sc.broadcast(conf.model)
        input_cols_bc = sc.broadcast(conf.input_cols)
        input_cols_type_bc = sc.broadcast(conf.input_cols_type)
        batch_size_bc = sc.broadcast(conf.batch_size)

        rawPredictionColBC = sc.broadcast(rawPredictionCol)
        predictionColBC = sc.broadcast(predictionCol)
        probabilityColBC = sc.broadcast(probabilityCol)

        def map_partitions_fn(rank, iterator):
            input_cols = input_cols_bc.value
            input_cols_type = input_cols_type_bc.value
            model: torch.nn.Module = model_bc.value
            model.eval()

            def collate_fn(rows):
                if input_cols_type:
                    res_dict = {name: torch.tensor([row[name] for row in rows], dtype=ty) for name, ty
                                in zip(input_cols, input_cols_type)}
                else:
                    res_dict = {name: torch.tensor([row[name] for row in rows]) for name in input_cols}
                res_dict['_row'] = [row.asDict() for row in rows]
                return res_dict

            dl = DataLoader(IterDataset(iterator), batch_size=batch_size_bc.value, collate_fn=collate_fn)

            with torch.no_grad():  # torch.set_grad_enabled(False)
                # send to device
                gpu_num = torch.cuda.device_count()
                if gpu_num:
                    device_rank = rank % gpu_num
                    device = torch.device(f"cuda:{device_rank}")
                else:
                    device = torch.device('cpu')
                model.to(device)

                for rows in dl:
                    try:
                        outputs = model(**{name: rows[name].to(device) for name in input_cols})
                    except Exception as e:
                        print(e)
                    else:
                        for rawPrediction, row in zip(outputs, rows['_row']):
                            if len(rawPrediction.shape):
                                prediction = float(torch.argmax(rawPrediction).item())
                                probability = DenseVector(torch.softmax(rawPrediction, dim=-1).tolist())
                                # prob = probability[prediction]
                            else:
                                probability = rawPrediction.item()
                                prediction = 1.0 if probability > 0.5 else 0.0
                                # prob = probability
                            row.update({rawPredictionColBC.value: DenseVector(rawPrediction.tolist()),
                                        predictionColBC.value: prediction,
                                        probabilityColBC.value: probability})
                            yield row
        return rdd.mapPartitionsWithIndex(map_partitions_fn)


class PytorchClassifyModel(Model, HasRawPredictionCol, HasProbabilityCol, HasPredictionCol, MLWritable, MLReadable):
    inputCols = Param(Params._dummy(), "inputCols", "input cols", typeConverter=TypeConverters.toListString)
    batchSize = Param(Params._dummy(), "batchSize", "batch size", typeConverter=TypeConverters.toInt)
    # inputColsType = Param(Params._dummy(), "inputColsType", "types of input cols", typeConverter=TypeConverters.toListString)

    def __init__(self, model: torch.nn.Module = None, input_cols: List[str] = None, batch_size: int = 100  #, input_cols_type: List[str] = None
                 ):
        super().__init__()
        self._setDefault(inputCols=['data'], batchSize=100)
        self._set(inputCols=input_cols, batchSize=batch_size  #, inputColsType=input_cols_type
                  )
        self.model = model

    def setBatchSize(self, value):
        return self._set(batchSize=value)

    def _transform(self, dataset: DataFrame):
        input_cols = self.getOrDefault(self.inputCols)
        spark = SparkSession.builder.getOrCreate()
        rdd = self.vector2array(dataset, input_cols).rdd

        master: str = spark.conf.get('spark.master', 'none')
        if master.startswith('local'):
            gpu_num = torch.cuda.device_count()
            if gpu_num:
                rdd = rdd.repartition(gpu_num)

        conf = Config(model=self.model, input_cols=input_cols,
                      batch_size=self.getOrDefault(self.batchSize)  # , input_cols_type=input_cols_type
                      )
        raw_prediction_col = self.getOrDefault(self.rawPredictionCol)
        prediction_col = self.getOrDefault(self.predictionCol)
        probability_col = self.getOrDefault(self.probabilityCol)
        rdd = LocalGPUDistTrain.prediction(rdd, conf, rawPredictionCol=raw_prediction_col,
                                           predictionCol=prediction_col,
                                           probabilityCol=probability_col)
        schema = dataset.schema.add(StructField(raw_prediction_col, VectorUDT(), False))\
            .add(StructField(prediction_col, DoubleType(), False)) \
            .add(StructField(probability_col, VectorUDT(), False))
        return spark.createDataFrame(rdd, schema=schema)

    def write(self):
        return self

    def save(self, path):
        DefaultParamsWriter(self).save(path)
        sc = SparkSession.builder.getOrCreate().sparkContext
        buffer = io.BytesIO()
        torch.save(self.model, buffer)
        sc.parallelize([buffer.getvalue()], 1).saveAsPickleFile(f'{path}/model.pk')

    @classmethod
    def read(cls):
        return cls

    @classmethod
    def load(cls, path):
        m: PytorchClassifyModel = DefaultParamsReader(cls).load(path)
        sc = SparkSession.builder.getOrCreate().sparkContext
        model_pk_str = sc.pickleFile(f'{path}/model.pk', 1).collect()[0]
        buffer = io.BytesIO(model_pk_str)
        m.model = torch.load(buffer)
        return m

    @staticmethod
    def vector2array(dataset: DataFrame, input_cols: list):
        vector_type = VectorUDT()
        names = [s.name for s in dataset.schema if
                 s.dataType == vector_type and s.name in input_cols]
        for name in names:
            dataset = dataset.withColumn(name, udf_vec2arr(name))
        return dataset


class PytorchLocalGPUClassifier(Estimator):

    @keyword_only
    def __init__(self, model=None, optimizer=None, criterion=None, buffer_size=500, batch_size=50, epochs=1,
                 input_cols=None, input_cols_type=None,
                 label_col='label', label_col_type=None,
                 checkpoint_path='./model.checkpoint', print_every=100,
                 early_stop=10000):
        super().__init__()
        kwargs = self._input_kwargs
        self.conf = Config(**kwargs)

    @classmethod
    def from_conf(cls, conf: Config = None):
        m = cls()
        m.conf = conf
        return m

    def _fit(self, dataset: DataFrame) -> PytorchClassifyModel:
        dataset = PytorchClassifyModel.vector2array(dataset, self.conf.input_cols)
        LocalGPUDistTrain.train(dataset.rdd, self.conf)
        return PytorchClassifyModel(model=self.conf.model, input_cols=self.conf.input_cols,
                                    batch_size=self.conf.batch_size)

import torch
import random
from torch.utils.data import IterableDataset
from typing import Iterator, Callable, Iterable
from itertools import tee, chain


class IterDataset(IterableDataset):
    def __init__(self, iterator: Iterator, tee_n=1):
        super().__init__()
        if tee_n > 1:
            self.iterator = chain(*tee(iterator, tee_n))
        else:
            self.iterator = iterator

    def __iter__(self):
        try:
            while True:
                try:
                    item = next(self.iterator)
                except StopIteration:
                    break
                else:
                    yield item
        except GeneratorExit:
            pass


class FlatMapIterDataset(IterableDataset):
    def __init__(self, iterator: Iterator, flat_map_fn: Callable[..., Iterable], tee_n=1):
        super().__init__()
        if tee_n > 1:
            self.iterator = chain(*tee(iterator, tee_n))
        else:
            self.iterator = iterator
        self.flat_map_fn = flat_map_fn

    def __iter__(self):
        try:
            while True:
                try:
                    items = next(self.iterator)
                except StopIteration:
                    break
                else:
                    for i in self.flat_map_fn(items):
                        yield i
        except GeneratorExit:
            pass


# class ShuffleDataset(IterableDataset):
#     def __init__(self, iterator: iter, buffer_size=500, tee_n=1):
#         super().__init__()
#         if tee_n > 1:
#             self.iterator = chain(*tee(iterator, tee_n))
#         else:
#             self.iterator = iterator
#         self.buffer_size = buffer_size
#
#     def __iter__(self):
#         if self.buffer_size <= 1:
#             return self.iterator
#         # build buffer
#         buf = []
#         try:
#             for i in range(self.buffer_size):
#                 buf.append(next(self.iterator))
#         except:
#             self.buffer_size = len(buf)
#         # random in buffer
#         try:
#             while True:
#                 try:
#                     item = next(self.iterator)
#                     evict_idx = random.randint(0, self.buffer_size - 1)
#                     yield buf[evict_idx]
#                     buf[evict_idx] = item
#                 except StopIteration:
#                     break
#             while len(buf) > 0:
#                 yield buf.pop()
#         except GeneratorExit:
#             pass


class FlatMapShuffleDataset(IterableDataset):
    def __init__(self, iterator: Iterator, flat_map_fn: Callable[..., Iterable],
                 buffer_size=2000, next_buf_nums=10, tee_n=1):
        super().__init__()
        if tee_n > 1:
            self.iterator = chain(*tee(iterator, tee_n))
        else:
            self.iterator = iterator
        self.flat_map_fn = flat_map_fn
        self.buffer_size = buffer_size
        self.next_buf_nums = next_buf_nums

    def __iter__(self):
        if self.buffer_size <= 1:
            return iter(FlatMapIterDataset(self.iterator, self.flat_map_fn, tee_n=1))
        buf = []
        next_buf = []
        next_buf_i = 0
        # build
        while len(buf) < self.buffer_size:
            try:
                items = next(self.iterator)
            except StopIteration:
                break
            else:
                buf += list(self.flat_map_fn(items))
        self.buffer_size = len(buf)
        # random in buf
        try:
            while True:
                if next_buf_i == 0:
                    # next_buf
                    next_buf = []
                    for i in range(self.next_buf_nums):
                        try:
                            items = next(self.iterator)
                        except StopIteration:
                            break
                        else:
                            next_buf += list(self.flat_map_fn(items))
                    next_buf_i = len(next_buf)
                    if next_buf_i == 0:
                        break
                # evict
                evict_idx = random.randint(0, self.buffer_size - 1)
                yield buf[evict_idx]
                next_buf_i -= 1
                buf[evict_idx] = next_buf[next_buf_i]
            while len(buf) > 0:
                yield buf.pop()
        except GeneratorExit:
            pass


# class ModelPredictionIter:
#
#     def __init__(self, rdd_partition_iter, model, batch_size=200):
#         self.iter = rdd_partition_iter
#         self.batch_size = batch_size
#         self.model = model
#         self.iterHasNext = True
#         self.batches = iter([])
#
#     def map_partition_func(self, rows) -> iter:
#         lines = torch.stack([torch.tensor(row.lines) for row in rows])
#         aids = torch.stack([torch.tensor(row.aid) for row in rows])
#         self.model.eval()
#         with torch.no_grad():
#             outputs = self.model(lines, aids)
#         prediction = [1 if out >= 0 else 0 for out in outputs]
#         probability = (outputs + 1) / 2
#
#         def zipout(zipobj):
#             rowdict, out, outs = zipobj
#             rowdict['probability'] = outs
#             rowdict['prediction'] = out
#             return rowdict
#
#         return map(zipout, zip(rows, prediction, probability))
#
#     def __batch_calculate(self):
#         it = 0
#         datas = []
#         for row in self.iter:
#             it += 1
#             datas.append(row.asDict())
#             if it >= self.batch_size:
#                 break
#         else:
#             self.iterHasNext = False
#         if datas:
#             self.batches = self.map_partition_func(datas)
#
#     def __iter__(self):
#         return self
#
#     def __next__(self):
#         try:
#             data = next(self.batches)
#         except StopIteration:
#             if self.iterHasNext:
#                 self.__batch_calculate()
#                 data = next(self.batches)
#             else:
#                 raise StopIteration
#         return data

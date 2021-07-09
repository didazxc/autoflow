import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pyspark.sql import Row
from typing import List


def eval_binary(arr: List[Row], label_col: str, categories=None):
    if categories is None:
        categories = [0, 1]
    print(f'first_line: {arr[0]}')
    # auc
    eval_roc_fig(arr, labelCol=label_col, rawPredictionCol='rawPrediction')
    # acc
    eval_confusion_fig(arr, categories, label_index=label_col, prediction_index='prediction')


def eval_roc(samples: List[Row], n_bins=1000, labelCol='label', rawPredictionCol='rawPrediction'):
    """
        samples is a iterable object, cate=sample[cate_index], probability=sample[probability_index],
        so samples should has __getitem__.
        probability is the probability of cate=pos_label.
    """
    pos = [.0] * n_bins
    neg = [.0] * n_bins
    m = 0
    n = 0
    p_s = [(s[rawPredictionCol][1], s[labelCol] == 1.0) for s in samples]
    p_min = min(p_s,key=lambda x:x[0])[0]
    p_max = max(p_s,key=lambda x:x[0])[0]
    for s in p_s:
        p = (s[0]-p_min)/(p_max-p_min)
        idx = int(p * n_bins) if p < 1.0 else n_bins - 1
        if s[1]:
            pos[idx] += 1
            m += 1
        else:
            neg[idx] += 1
            n += 1
    pos = list(map(lambda x: x / m, pos))
    neg = list(map(lambda x: x / n, neg))
    auc = pos[0] * neg[0] * 0.5
    accumulated_neg = 0
    for i in range(1, n_bins):
        accumulated_neg += neg[i - 1]
        auc += pos[i] * (accumulated_neg + neg[i] * 0.5)
    for i in range(n_bins - 2, -1, -1):
        pos[i] = pos[i + 1] + pos[i]
        neg[i] = neg[i + 1] + neg[i]
    return auc, pos, neg


def eval_roc_fig(samples: List[Row], n_bins=1000, labelCol='label', rawPredictionCol='rawPrediction'):
    auc, pos, neg = eval_roc(samples, n_bins=n_bins,
                             labelCol=labelCol, rawPredictionCol=rawPredictionCol)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(neg, pos)
    ax.fill_between(neg, pos, interpolate=True, color='green', alpha=0.5)
    ax.text(0.8, 0.2, f'AUC: {auc: .2f}')
    ax.set(xlim=[0, 1], ylim=[0, 1])
    plt.show()


def eval_auc(samples: List[Row], labelCol='label', rawPredictionCol='rawPrediction'):
    samples.sort(key=lambda x: x[rawPredictionCol][1])
    rank = 0
    m = 0
    n = 0
    for i, s in enumerate(samples):
        if s[labelCol] == 1.0:
            m += 1
            rank += i
        else:
            n += 1
    return (rank - m * (m - 1) / 2) / (m * n)


def eval_confusion(samples: List[Row], categories_num, label_index='label', prediction_index='prediction'):
    n = categories_num
    confusion = np.zeros((n, n))
    for sample in samples:
        confusion[int(sample[label_index])][int(sample[prediction_index])] += 1
    acc = sum(confusion[i][i] for i in range(n)) / len(samples)
    for i in range(n):
        confusion[i] = confusion[i] / confusion[i].sum()
    return acc, confusion


def eval_confusion_fig(samples: List[Row], categories, label_index='label', prediction_index='prediction'):
    acc, confusion = eval_confusion(samples=samples, categories_num=len(categories),
                                    label_index=label_index, prediction_index=prediction_index)
    # plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(confusion)
    fig.colorbar(cax)
    labels = [''] + categories
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.text(0, 0, f'Rc: {confusion[0][0]: .2f}')
    ax.text(0, 1, f'Rc: {confusion[1][0]: .2f}')
    ax.text(1, 0, f'Rc: {confusion[0][1]: .2f}')
    ax.text(1, 1, f'Rc: {confusion[1][1]: .2f}')
    ax.set_title(f'ACC:{acc:.2f}')
    plt.show()


# ！-*- coding:utf-8 -*-
import random

import numpy as np


def clipDataTopX(dataToClip, top=3):
    res = [sorted(s, reverse=True)[0:top] for s in dataToClip]
    return np.array(res)


def shuffleAndSplitData(dataX, dataY, cluster):
    # dataX = np.concatenate((dataX, testX), axis = 0)
    # dataY = np.concatenate((dataY, testY), axis = 0)

   # shuffle data
    c = list(zip(dataX, dataY))
    random.shuffle(c)
    dataX, dataY = zip(*c)


    # cluster_10000 = int(cluster / 2 + cluster)

    toTrainData = np.array(dataX[:cluster])
    toTrainLabel = np.array(dataY[:cluster])

    toTestData = np.array(dataX[cluster:cluster * 2])
    toTestLabel = np.array(dataY[cluster:cluster * 2])
    # toTestData = np.array(dataX[cluster:cluster_10000])
    # toTestLabel = np.array(dataY[cluster:cluster_10000])

    # cluster_10000_ = cluster_10000 + cluster
    # shadowData = np.array(dataX[cluster_10000:cluster_10000_])
    # shadowLabel = np.array(dataY[cluster_10000:cluster_10000_])

    shadowData = np.array(dataX[cluster * 2:cluster * 3])
    shadowLabel = np.array(dataY[cluster * 2:cluster * 3])

    shadowTestData = np.array(dataX[cluster * 3:cluster * 4])
    shadowTestLabel = np.array(dataY[cluster * 3:cluster * 4])
    # shadowTestData = np.array(dataX[cluster_10000_:cluster*3])
    # shadowTestLabel = np.array(dataY[cluster_10000_:cluster*3])

    return toTrainData, toTrainLabel, shadowData, shadowLabel, toTestData, toTestLabel, shadowTestData, shadowTestLabel




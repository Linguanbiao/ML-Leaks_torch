import os

import numpy as np
import random
import pickle
from torchvision import transforms
from torchvision.datasets import MNIST, CIFAR10, CIFAR100
def preprocessingCIFAR(toTrainData, toTestData):
    def reshape_for_save(raw_data):
        raw_data = np.dstack(
            (raw_data[:, :1024], raw_data[:, 1024:2048], raw_data[:, 2048:]))
        raw_data = raw_data.reshape(
            (raw_data.shape[0], 32, 32, 3)).transpose(0, 3, 1, 2)
        return raw_data.astype(np.float32)

    offset = np.mean(reshape_for_save(toTrainData), 0)
    scale = np.std(reshape_for_save(toTrainData), 0).clip(min=1)

    def rescale(raw_data):
        return (reshape_for_save(raw_data) - offset) / scale

    return rescale(toTrainData), rescale(toTestData)

def load_cifar10(data_folder):
    for i in range(5):
        f = open(data_folder + '/data_batch_' + str(i + 1), 'rb')

        train_data_dict = pickle.load(f, encoding='iso-8859-1')

        f.close()
        if i == 0:
            X = train_data_dict["data"]
            y = train_data_dict["labels"]
            continue
        X = np.concatenate((X, train_data_dict["data"]), axis=0)
        y = np.concatenate((y, train_data_dict["labels"]), axis=0)
    f = open(data_folder + '/test_batch', 'rb')
    test_data_dict = pickle.load(f, encoding='iso-8859-1')
    f.close()
    XTest = np.array(test_data_dict["data"])
    yTest = np.array(test_data_dict["labels"])
    dataX = np.concatenate((X, XTest), axis = 0)
    dataY = np.concatenate((y, yTest), axis = 0)
    c = list(zip(dataX, dataY))
    random.shuffle(c)
    dataX, dataY = zip(*c)
    return dataX, dataY

# dataX, dataY = load_cifar10("../data/cifar-10-batches-py")
# TTX, TTy, TTX_test, TTy_test = np.array(dataX[:10000]), np.array(dataY[:10000]), np.array(dataX[10000:20000]), np.array(dataY[10000:20000])
# TTX, TTX_test = preprocessingCIFAR(TTX, TTX_test)
# np.savez('./teacher_targetTrain_Test_data.npz', TTX, TTy, TTX_test, TTy_test)
# STX, STy, STX_test, STy_test = np.array(dataX[20000:30000]), np.array(dataY[20000:30000]), np.array(dataX[30000:40000]), np.array(dataY[30000:40000])
# STX, STX_test = preprocessingCIFAR(STX, STX_test)
# np.savez('./student_targetTrain_Test_data.npz', STX, STy, STX_test, STy_test)
# SX, Sy, SX_test, Sy_test = np.array(dataX[40000:50000]), np.array(dataY[40000:50000]), np.array(dataX[50000:60000]), np.array(dataY[50000:60000])
# SX, SX_test = preprocessingCIFAR(SX, SX_test)
# np.savez('./shadowTrain_Test_data.npz', SX, Sy, SX_test, Sy_test)

def load_Purchase(data_folder):
    data = np.load(data_folder)
    features = data['features']  # (197324,600)
    labels = data['labels']  # (197324,)
    X = features
    y = labels
    c = list(zip(X, y))
    random.shuffle(c)
    X, y = zip(*c)
    X = np.array(X)
    y = np.array(y)
    return X, y

def load_Purchase100():
    #x = pickle.load(open('F:/Chen/LASTEST/ML-Leaks/data/purchase_100/purchase_100_n_features.p', 'rb'))
    #y = pickle.load(open('F:/Chen/LASTEST/ML-Leaks/data/purchase_100/purchase_100_n_labels.p', 'rb'))
    #x = np.array(x, dtype=np.float32)
    #y = np.array(y, dtype=np.int32)
    #print(x.shape)
    #print(y.shape)
    #x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.37, random_state=0)
    #print(x_train.shape)
    #print(x_test.shape)
    #print(y_train.shape)
    #print(y_test.shape)

    DATASET_PATH='.\purchase'

    DATASET_NAME= 'dataset_purchase'

    DATASET_FILE = os.path.join(DATASET_PATH,DATASET_NAME)

    data_set =np.genfromtxt(DATASET_FILE,delimiter=',')

    X = data_set[:,1:].astype(np.float64)
    y = (data_set[:,0]).astype(np.int32)-1
    return X, y

# dataX, dataY = load_Purchase("./purchase/purchase2.npz")
# dataX, dataY = load_Purchase100()
# TTX, TTy, TTX_test, TTy_test = dataX[:10000], dataY[:10000], dataX[10000:20000], dataY[10000:20000]
# np.savez('./teacher_targetTrain_Test_data.npz', TTX, TTy, TTX_test, TTy_test)
# STX, STy, STX_test, STy_test = dataX[20000:30000], dataY[20000:30000], dataX[30000:40000], dataY[30000:40000]
# np.savez('./student_targetTrain_Test_data.npz', STX, STy, STX_test, STy_test)
# SX, Sy, SX_test, Sy_test = dataX[40000:50000], dataY[40000:50000], dataX[50000:60000], dataY[50000:60000]
# np.savez('./shadowTrain_Test_data.npz', SX, Sy, SX_test, Sy_test)

# 先检测是否有cifar100数据，没有则下载
def readCIFAR100(data_path):
    trainset = CIFAR100(data_path, train = True, download=True)
    testset = CIFAR100(data_path, train = False, download = True)

    trainX, trainY = trainset.data, trainset.targets
    testX, testY = testset.data, testset.targets

    for matrix in trainX:
        matrix.transpose(0, 2, 1)

    for matrix in testX:
        matrix.transpose(0, 2, 1)

    trainX = trainX.reshape(trainX.shape[0], -1)
    testX = testX.reshape(testX.shape[0], -1)

    dataX = np.concatenate((trainX, testX), axis = 0)
    dataY = np.concatenate((trainY, testY), axis = 0)
    c = list(zip(dataX, dataY))
    random.shuffle(c)
    dataX, dataY = zip(*c)
    dataX = np.array(dataX)
    dataY = np.array(dataY)
    return dataX, dataY

def load_cifar100(data_folder):
    f = open(data_folder + '/train', 'rb')
    train_data_dict = pickle.load(f, encoding='iso-8859-1')
    f.close()

    X = train_data_dict['data']
    y = train_data_dict['fine_labels']

    f = open(data_folder + '/test', 'rb')
    test_data_dict = pickle.load(f, encoding='iso-8859-1')
    f.close()

    XTest = np.array(test_data_dict['data'])
    yTest = np.array(test_data_dict['fine_labels'])

    dataX = np.concatenate((X, XTest), axis=0)
    dataY = np.concatenate((y, yTest), axis=0)
    c = list(zip(dataX, dataY))
    random.shuffle(c)
    dataX, dataY = zip(*c)
    return dataX, dataY

# dataX, dataY = readCIFAR100('../data')
dataX, dataY = load_cifar100("../data/cifar-100-python")
TTX, TTy, TTX_test, TTy_test = np.array(dataX[:10000]), np.array(dataY[:10000]), np.array(dataX[10000:20000]), np.array(dataY[10000:20000])
TTX, TTX_test = preprocessingCIFAR(TTX, TTX_test)
np.savez('./teacher_targetTrain_Test_data.npz', TTX, TTy, TTX_test, TTy_test)
STX, STy, STX_test, STy_test = np.array(dataX[20000:30000]), np.array(dataY[20000:30000]), np.array(dataX[30000:40000]), np.array(dataY[30000:40000])
STX, STX_test = preprocessingCIFAR(STX, STX_test)
np.savez('./student_targetTrain_Test_data.npz', STX, STy, STX_test, STy_test)
SX, Sy, SX_test, Sy_test = np.array(dataX[40000:50000]), np.array(dataY[40000:50000]), np.array(dataX[50000:60000]), np.array(dataY[50000:60000])
SX, SX_test = preprocessingCIFAR(SX, SX_test)
np.savez('./shadowTrain_Test_data.npz', SX, Sy, SX_test, Sy_test)
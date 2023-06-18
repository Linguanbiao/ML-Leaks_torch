import random

import numpy as np
import torch
from sklearn import model_selection, datasets
from sklearn.metrics import accuracy_score, classification_report
from torch import optim

from dataset import readCIFAR10, readCIFAR100, readFLW,readMINST
from net.CNN import CNN_Model
from preprocessing import preprocessingCIFAR,preprocessingMINST
from train import CrossEntropy_L2, iterate_minibatches


def shuffleAndSplitData(dataX, dataY, cluster):
    c = list(zip(dataX, dataY))
    random.shuffle(c)
    dataX, dataY = zip(*c)

    toTrainData = np.array(dataX[:cluster])
    toTrainLabel = np.array(dataY[:cluster])


    toTestData = np.array(dataX[cluster:cluster * 2])
    toTestLabel = np.array(dataY[cluster:cluster * 2])


    cluster_1 = cluster * 2 + 1000
    shadowData = np.array(dataX[cluster*2 :cluster_1])
    shadowLabel = np.array(dataY[cluster*2 :cluster_1])


    # shadowTestData = np.array(dataX[cluster_1:-1])
    # shadowTestLabel = np.array(dataY[cluster_1:-1])
    shadowTestData = np.array(dataX[cluster:-1])
    shadowTestLabel = np.array(dataY[cluster:-1])

    return toTrainData, toTrainLabel, shadowData, shadowLabel, toTestData, toTestLabel, shadowTestData, shadowTestLabel


def read_cifar10(originalDatasetPath):
    print("Loading data")
    dataX, dataY, _, _ = readCIFAR10(originalDatasetPath)
    print("Preprocessing data")

    cluster =10520

    toTrainData, toTrainLabel, shadowData, shadowDataLabel, toTestData, toTestLabel, shadowTestData, shadowTestLabel = \
        shuffleAndSplitData(dataX, dataY, cluster)

    return toTrainData, toTrainLabel, toTestData, toTestLabel, shadowData, shadowDataLabel, shadowTestData, shadowTestLabel

def read_cifar100(originalDatasetPath):
    print("Loading data")
    dataX, dataY, _, _ = readCIFAR100(originalDatasetPath)
    print("Preprocessing data")

    cluster = 10500

    toTrainData, toTrainLabel, shadowData, shadowDataLabel, toTestData, toTestLabel, shadowTestData, shadowTestLabel = \
        shuffleAndSplitData(dataX, dataY, cluster)

    return toTrainData, toTrainLabel, toTestData, toTestLabel, shadowData, shadowDataLabel, shadowTestData, shadowTestLabel

def read_MINST(originalDatasetPath):
    print("Loading data")
    dataX, dataY, _, _ = readMINST(originalDatasetPath)
    print("Preprocessing data")

    cluster = 10520

    toTrainData, toTrainLabel, shadowData, shadowDataLabel, toTestData, toTestLabel, shadowTestData, shadowTestLabel = \
        shuffleAndSplitData(dataX, dataY, cluster)

    return toTrainData, toTrainLabel, toTestData, toTestLabel, shadowData, shadowDataLabel, shadowTestData, shadowTestLabel

def read_FLW(originalDatasetPath):

    print("Loading data")
    dataX, dataY, _, _ = readFLW(originalDatasetPath)
    print("Preprocessing data")

    cluster =300

    toTrainData, toTrainLabel, shadowData, shadowDataLabel, toTestData, toTestLabel, shadowTestData, shadowTestLabel = \
        shuffleAndSplitData(dataX, dataY, cluster)

    return toTrainData, toTrainLabel, toTestData, toTestLabel, shadowData, shadowDataLabel, shadowTestData, shadowTestLabel

def train(toTrainData,
          toTrainLabel,
          toTestData,
          toTestLabel,
          shadowData,
          shadowDataLabel,
          n_hidden=50,
          batch_size=100,
          epochs=50,
          learning_rate=0.01,
          model='cnn',
          l2_ratio=1e-7):
    toTrainData = toTrainData.astype(np.float32)
    toTrainLabel = toTrainLabel.astype(np.int32)
    toTestData = toTestData.astype(np.float32)
    toTestLabel = toTestLabel.astype(np.int32)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    n_in = toTrainData.shape
    n_out = len(np.unique(toTrainLabel))

    if batch_size > len(toTrainLabel):
        batch_size = len(toTrainLabel)

    net = CNN_Model(n_in, n_hidden, n_out)
    net.to(device)

    m = n_in[0]
    criterion = CrossEntropy_L2(net, m, l2_ratio).to(device)

    optimizer = optim.Adam(net.parameters(), lr=learning_rate)

    print('Training...')
    net.train()

    temp_loss = 0.0

    for epoch in range(epochs):

        for input_batch, target_batch in iterate_minibatches(toTrainData, toTrainLabel, batch_size):
            input_batch, target_batch = torch.tensor(input_batch).contiguous(), torch.tensor(target_batch).type(torch.long)
            input_batch, target_batch = input_batch.to(device), target_batch.to(device)

            # empty parameters in optimizer
            optimizer.zero_grad()

            outputs = net(input_batch)
            # outputs [100, 10]

            # calculate loss value
            loss = criterion(outputs, target_batch)

            # back propagation
            loss.backward()

            # update paraeters in optimizer(update weight)
            optimizer.step()

            temp_loss += loss.item()

        temp_loss = round(temp_loss, 3)
        if epoch % 5 == 0:
            print('Epoch {}, train loss {}'.format(epoch, temp_loss))

        temp_loss = 0.0

    net.eval()

    pred_y = []

    with torch.no_grad():

        for input_batch, _ in iterate_minibatches(toTrainData, toTrainLabel, batch_size, shuffle=False):
            input_batch = torch.tensor(input_batch).contiguous()
            input_batch = input_batch.to(device)

            outputs = net(input_batch)

            pred_y.append(torch.max(outputs, 1)[1].data.cpu().numpy())

        pred_y = np.concatenate(pred_y)

    print('Train Accuracy: {}'.format(accuracy_score(toTrainLabel, pred_y)))

    pred_y = []

    with torch.no_grad():

        for input_batch, _ in iterate_minibatches(toTestData, toTestLabel, batch_size, shuffle=False):
            input_batch = torch.tensor(input_batch).contiguous()
            input_batch = input_batch.to(device)

            outputs = net(input_batch)

            pred_y.append(torch.max(outputs, 1)[1].data.cpu().numpy())

        pred_y = np.concatenate(pred_y)

    print('Test Accuracy: {}'.format(accuracy_score(toTestLabel, pred_y)))


    # 寻找阈值

    generate_data = shadowData[:1000]
    generate_data_ = shadowDataLabel[:1000]

    # count = dict()
    #
    # for label in generate_data_:
    #
    #     if label in count:
    #         count[label] += 1
    #     else:
    #         count[label] = 1
    #
    # for key in count:
    #     print('Key : ', key, ' Number : ', count[key], '\n')

    net.eval()


    confidence = []

    with torch.no_grad():

        for input_batch, _ in iterate_minibatches(generate_data, generate_data_, batch_size, shuffle=False):
            input_batch = torch.tensor(input_batch).contiguous()
            input_batch = input_batch.to(device)

            outputs = net(input_batch)
            outputs_ = torch.nn.functional.softmax(outputs, dim=1)

            column = torch.max(outputs, 1)[1]
            row = [i for i in range(batch_size)]
            confidence.append(outputs_[row, column].detach().cpu().numpy())

        confidence = np.concatenate(confidence)

    confidence.sort()
    confidence = abs(np.sort(-confidence))


    rate = 0.1
    destination_index = int(len(confidence) * rate)
    print(confidence[destination_index])

    # t = 0.98
    print(rate)
    # print(t)

    label_1, label_0 = np.ones(len(toTrainLabel)), np.zeros(len(toTestLabel))
    # label = np.concatenate((label_1, label_0), axis=0)

    # toTrainData = np.concatenate((toTrainData, toTestData), axis=0)
    # toTrainLabel = np.concatenate((toTrainLabel, toTestLabel), axis=0)

    net.eval()

    pred_y = []

    with torch.no_grad():

        for input_batch, _ in iterate_minibatches(toTrainData, toTrainLabel, batch_size, shuffle=False):
            input_batch = torch.tensor(input_batch).contiguous()
            input_batch = input_batch.to(device)

            outputs = net(input_batch)
            outputs_ = torch.nn.functional.softmax(outputs, dim=1)

            column = torch.max(outputs, 1)[1].data.cpu().numpy()
            row = [i for i in range(batch_size)]
            pos = list(zip(row, column))

            for (x, y) in pos:
                if outputs_[x, y].detach().cpu().numpy() > confidence[destination_index]:
                    pred_y.append(1)
                else:
                    pred_y.append(0)


    print('chose Threshold train Accuracy: {}'.format(accuracy_score(label_1, pred_y)))

    net.eval()

    pred_y = []

    with torch.no_grad():

        for input_batch, _ in iterate_minibatches(toTestData, toTestLabel, batch_size, shuffle=False):
            input_batch = torch.tensor(input_batch).contiguous()
            input_batch = input_batch.to(device)

            outputs = net(input_batch)
            outputs_ = torch.nn.functional.softmax(outputs, dim=1)

            column = torch.max(outputs, 1)[1].data.cpu().numpy()
            row = [i for i in range(batch_size)]
            pos = list(zip(row, column))

            for (x, y) in pos:
                if outputs_[x, y].detach().cpu().numpy() > confidence[destination_index]:
                    pred_y.append(1)
                else:
                    pred_y.append(0)
    print('chose Threshold test Accuracy: {}'.format(accuracy_score(label_0, pred_y)))


    label = np.concatenate((label_1, label_0), axis=0)
    toTrainData = np.concatenate((toTrainData, toTestData), axis=0)
    toTrainLabel = np.concatenate((toTrainLabel, toTestLabel), axis=0)

    net.eval()

    pred_y = []

    with torch.no_grad():

        for input_batch, _ in iterate_minibatches(toTrainData, toTrainLabel, batch_size, shuffle=False):
            input_batch = torch.tensor(input_batch).contiguous()
            input_batch = input_batch.to(device)

            outputs = net(input_batch)
            outputs_ = torch.nn.functional.softmax(outputs, dim=1)

            column = torch.max(outputs, 1)[1].data.cpu().numpy()
            row = [i for i in range(batch_size)]
            pos = list(zip(row, column))

            for (x, y) in pos:
                if outputs_[x, y].detach().cpu().numpy() > confidence[destination_index]:
                    pred_y.append(1)
                else:
                    pred_y.append(0)
    print('attack Accuracy: {}'.format(accuracy_score(label, pred_y)))
    print('More detailed results:')
    print(classification_report(label, pred_y))



if __name__ == '__main__':
    originalDatasetPath = '../data/cifar-10-batches-py'
    toTrainData, toTrainLabel, toTestData, toTestLabel, shadowData, shadowDataLabel, shadowTestData, shadowTestLabel = \
        read_cifar10(originalDatasetPath)

    # toTrainData, toTestData = preprocessingMINST(toTrainData, toTestData)
    # shadowData, shadowTestData = preprocessingMINST(shadowData, shadowTestData)
    toTrainData, toTestData = preprocessingCIFAR(toTrainData, toTestData)
    shadowData, shadowTestData = preprocessingCIFAR(shadowData, shadowTestData)

    train(toTrainData=toTrainData,
          toTrainLabel=toTrainLabel,
          toTestData=toTestData,
          toTestLabel=toTestLabel,
          shadowData=shadowData,
          shadowDataLabel=shadowDataLabel,
          epochs=50,
          n_hidden=128,
          l2_ratio=1e-07,
          learning_rate=0.001)

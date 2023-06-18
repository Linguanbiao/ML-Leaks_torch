import argparse
import random

import numpy as np
import torch
from sklearn import model_selection, datasets
from sklearn.metrics import accuracy_score, classification_report
from torch import optim, nn

from dataset import readCIFAR10, readCIFAR100, readFLW, readMINST
from net.CNN import CNN_Model
from paint_distribution import paint_histogram, paint_scatter
from preprocessing import preprocessingCIFAR, preprocessingMINST
from train import CrossEntropy_L2, iterate_minibatches
from pyvacy import optim, analysis
from data_partition import clipDataTopX
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torch.utils.data.distributed
from opacus import PrivacyEngine


def shuffle(dataX, dataY):
    c = list(zip(dataX, dataY))
    random.shuffle(c)
    dataX, dataY = zip(*c)

    return np.array(dataX), np.array(dataY)


def read_cifar10(originalDatasetPath):
    print("Loading data")
    trainX, trainY, testX, testY = readCIFAR10(originalDatasetPath)
    print("Preprocessing data")
    trainX, trainY = shuffle(trainX, trainY)
    testX, testY = shuffle(testX, testY)

    return trainX, trainY, testX, testY


def train(toTrainData,
          toTrainLabel,
          toTestData,
          toTestLabel,
          epochs=30,
          n_hidden=50,
          batch_size=250,
          learning_rate=0.15,
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

    #criterion = CrossEntropy_L2(net, m, l2_ratio).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)

    visual_batch_size = 250
    visual_batch_rate = int(visual_batch_size / batch_size)

    privacy_engine = PrivacyEngine(
        net,
        batch_size=visual_batch_size,  # batch_size=256    epoch=90  lr=0.001
        sample_size=len(toTrainLabel),
        alphas=[1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64)),
        noise_multiplier=1.1,  # 1.1
        max_grad_norm=1,  # 1.0
        secure_rng=False,
        target_delta=1e-5
    )
    privacy_engine.attach(optimizer)

    print('Training...')
    net.train()

    temp_loss = 0.0
    top1_acc = []
    losses = []

    for epoch in range(epochs):

        for i, (input_batch, target_batch) in enumerate(iterate_minibatches(toTrainData, toTrainLabel, batch_size)):

            input_batch, target_batch = torch.tensor(input_batch).contiguous(), torch.tensor(target_batch).type(
                torch.long)
            input_batch, target_batch = input_batch.to(device), target_batch.to(device)

            # empty parameters in optimizer
            # optimizer.zero_grad()

            outputs = net(input_batch)
            # outputs [100, 10]

            # calculate loss value
            loss = criterion(outputs, target_batch)
            preds = np.argmax(outputs.detach().cpu().numpy(), axis=1)
            acc1 = (preds == target_batch.detach().cpu().numpy()).mean()
            losses.append(loss.item())
            top1_acc.append(acc1)

            # back propagation
            loss.backward()

            # update paraeters in optimizer(update weight)
            if ((i + 1) % visual_batch_rate == 0) or ((i + 1) == int(len(toTrainLabel) / batch_size)):
                optimizer.step()
            else:
                optimizer.virtual_step()

            temp_loss += loss.item()

            if (i + 1) % 50 == 0:
                epsilon, best_alpha = optimizer.privacy_engine.get_privacy_spent(1e-5)
                print(
                    f"\tTrain Epoch: {epoch} \t"
                    f"Loss: {np.mean(losses):.6f} "
                    f"Acc@1: {np.mean(top1_acc):.6f} "
                    f"(ε = {epsilon:.2f}, δ = {1e-5}) for α = {best_alpha}"
                )

        temp_loss = 0.0

    pred_y = []
    confidence_train = []

    # 从50000的训练集中抽10000个出来测试
    toTrainData = np.array(toTrainData[:10000])
    toTrainLabel = np.array(toTrainLabel[:10000])

    with torch.no_grad():

        for input_batch, _ in iterate_minibatches(toTrainData, toTrainLabel, batch_size, shuffle=False):
            input_batch = torch.tensor(input_batch).contiguous()
            input_batch = input_batch.to(device)

            outputs = net(input_batch)
            confidence = nn.functional.softmax(outputs)
            confidence = clipDataTopX(confidence.detach().cpu().numpy(), top=1)
            confidence_train.append(confidence)
            pred_y.append(torch.max(outputs, 1)[1].data.cpu().numpy())

        pred_y = np.concatenate(pred_y)
        confidence_train = np.concatenate(confidence_train, axis=0)

    print('Train Accuracy: {}'.format(accuracy_score(toTrainLabel, pred_y)))

    pred_y = []
    confidence_test = []
    with torch.no_grad():

        for input_batch, _ in iterate_minibatches(toTestData, toTestLabel, batch_size, shuffle=False):
            input_batch = torch.tensor(input_batch).contiguous()
            input_batch = input_batch.to(device)

            outputs = net(input_batch)
            confidence = nn.functional.softmax(outputs)
            confidence = clipDataTopX(confidence.detach().cpu().numpy(), top=1)
            confidence_test.append(confidence)
            pred_y.append(torch.max(outputs, 1)[1].data.cpu().numpy())

        pred_y = np.concatenate(pred_y)
        confidence_test = np.concatenate(confidence_test, axis=0)

    print('Test Accuracy: {}'.format(accuracy_score(toTestLabel, pred_y)))
    paint_histogram(confidence_train, confidence_test)
    # paint_scatter(confidence_train, confidence_test)


if __name__ == '__main__':

    originalDatasetPath = './data/cifar-10-batches-py'
    toTraindata, toTrainlabel, toTestdata, toTestlabel = \
        read_cifar10(originalDatasetPath)

    toTraindata, toTestdata = preprocessingCIFAR(toTraindata, toTestdata)

    train(toTrainData=toTraindata,
          toTrainLabel=toTrainlabel,
          toTestData=toTestdata,
          toTestLabel=toTestlabel,
          epochs=30,
          batch_size=250,
          n_hidden=128,
          l2_ratio=1e-07,
          learning_rate=0.001)

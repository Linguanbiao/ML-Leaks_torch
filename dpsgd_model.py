import argparse
import random
import numpy as np
import torch
import paint_distribution
from sklearn.metrics import accuracy_score
from torch import optim
from dataset import readCIFAR10
from net.CNN import CNN_Model
from preprocessing import preprocessingCIFAR
from train import CrossEntropy_L2, iterate_minibatches
from pyvacy import optim, analysis
from torch import nn
from data_partition import clipDataTopX


def shuffle(dataX, dataY):
    c = list(zip(dataX, dataY))
    # 将序列的所有元素随机排序
    random.shuffle(c)
    # zip(*c)与 zip 相反，可理解为解压，为zip的逆过程，可用于矩阵的转置
    dataX, dataY = zip(*c)
    return  np.array(dataX), np.array(dataY)


def read_cifar10(originalDatasetPath):
    print("Loading data")
    trainX, trainY, testX, testY = readCIFAR10(originalDatasetPath)
    print("Preprocessing data")
    trainX,trainY=shuffle(trainX,trainY)
    testX,testY=shuffle(testX,testY)

    return trainX,trainY,testX,testY


def train(toTrainData,
          toTrainLabel,
          toTestData,
          toTestLabel,
          n_hidden=50,
          batch_size=250,
          epochs=30,
          learning_rate=0.001,
          model='cnn',
          l2_ratio=1e-7):
    toTrainData = toTrainData.astype(np.float32)
    toTrainLabel = toTrainLabel.astype(np.int32)
    toTestData = toTestData.astype(np.float32)
    toTestLabel = toTestLabel.astype(np.int32)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    n_in = toTrainData.shape
    # np.unique()该函数是去除数组中的重复数字，并进行排序之后输出
    n_out = len(np.unique(toTrainLabel))

    if batch_size > len(toTrainLabel):
        batch_size = len(toTrainLabel)

    net = CNN_Model(n_in, n_hidden, n_out)
    # 将所有最开始读取数据时的tensor变量copy一份到device所指定的GPU上去，之后的运算都在GPU上进行。
    net.to(device)

    m = n_in[0]
    criterion = CrossEntropy_L2(net, m, l2_ratio).to(device)

    # optimizer = optim.Adam(net.parameters(), lr=learning_rate)

    optimizer = optim.DPSGD(
        l2_norm_clip=1.0, # 1.0
        noise_multiplier=1.1, # 1.1
        minibatch_size=250,
        microbatch_size=250,
        lr=learning_rate,
        momentum=0,
        params=net.parameters()
    )

    print('Achieves ({}, {})-DP'.format(
        analysis.moments_accountant(
            N=50000,
            batch_size=250,
            noise_multiplier =1.1, # 1.1
            epochs = 30,
            delta = 1e-5
        ),
        params['delta'],
    ))

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
            # 计算损失值
            temp_loss += loss.item()

        # 对所求的损失值保留三位小数
        temp_loss = round(temp_loss, 3)
        if epoch % 5 == 0:
            print('Epoch {}, train loss {}'.format(epoch, temp_loss))

        temp_loss = 0.0

    net.eval() # 把网络设定为训练状态

    pred_y = []
    # 用于保存训练集中置信度
    conference_train = []
    with torch.no_grad():
        for input_batch, _ in iterate_minibatches(toTrainData, toTrainLabel, batch_size, shuffle=False):
            input_batch = torch.tensor(input_batch).contiguous()
            input_batch = input_batch.to(device)
            outputs = net(input_batch)
            confidence = nn.functional.softmax(outputs)
            confidence = clipDataTopX(confidence.detach().cpu().numpy(), top=1)
            conference_train.append(confidence)
            pred_y.append(torch.max(outputs, 1)[1].data.cpu().numpy())

        pred_y = np.concatenate(pred_y)
        conference_train = np.concatenate(conference_train, axis=0)
    print('Train Accuracy: {}'.format(accuracy_score(toTrainLabel, pred_y)))

    pred_y = []
    # 用于保存测试集中置信度
    conference_test = []
    with torch.no_grad(): # 不需要进行网络参数的更新,也不需要进行反向传播
        for input_batch, _ in iterate_minibatches(toTestData, toTestLabel, batch_size, shuffle=False):
            input_batch = torch.tensor(input_batch).contiguous()
            input_batch = input_batch.to(device)
            outputs = net(input_batch)
            # 将张量的每个元素缩放到（0,1）区间且和为1
            confidence = nn.functional.softmax(outputs)
            # 将张量转化为numpy类型
            confidence = clipDataTopX(confidence.detach().cpu().numpy(), top=1)
            conference_test.append(confidence)
            pred_y.append(torch.max(outputs, 1)[1].data.cpu().numpy())

        pred_y = np.concatenate(pred_y)
        # 对conference_test按行生成一个列表
        conference_test = np.concatenate(conference_test,axis=0)

    print('Test Accuracy: {}'.format(accuracy_score(toTestLabel, pred_y)))
    # 绘图
    paint_distribution.paint_histogram(conference_train,conference_test)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--delta', type=float, default=1e-6, help='delta for epsilon calculation (default: 1e-5)')
    # parser.add_argument('--device', type=str, default=('cuda' if torch.cuda.is_available() else 'cpu'),
    #                     help='whether or not to use cuda (default: cuda if available)')
    # parser.add_argument('--iterations', type=int, default=14000, help='number of iterations to train (default: 14000)')
    # parser.add_argument('--l2-norm-clip', type=float, default=1.,
    #                     help='upper bound on the l2 norm of gradient updates (default: 0.1)')
    # parser.add_argument('--l2-penalty', type=float, default=0.001, help='l2 penalty on model weights (default: 0.001)')
    # parser.add_argument('--lr', type=float, default=0.15, help='learning rate (default: 0.15)')
    # parser.add_argument('--microbatch-size', type=int, default=1,
    #                     help='input microbatch size for training (default: 1)')
    # parser.add_argument('--minibatch-size', type=int, default=256,
    #                     help='input minibatch size for training (default: 256)')
    # parser.add_argument('--noise-multiplier', type=float, default=1.1,
    #                     help='ratio between clipping bound and std of noise applied to gradients (default: 1.1)')
    params = vars(parser.parse_args())

    originalDatasetPath = './data/cifar-10-batches-py'
    toTraindata,toTrainlabel,toTestdata,toTestlabel = \
        read_cifar10(originalDatasetPath)

    toTraindata, toTestdata = preprocessingCIFAR(toTraindata,toTestdata)


    train(toTrainData=toTraindata,
          toTrainLabel=toTrainlabel,
          toTestData=toTestdata,
          toTestLabel=toTestlabel,
          epochs=30,
          n_hidden=128,
          l2_ratio=1e-07,
          model='cnn',
          learning_rate= 0.001)   # 0.001



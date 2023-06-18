import random
import numpy as np
import torch
import  pandas as pd
from opacus import PrivacyEngine
from sklearn.model_selection import train_test_split
from torch import optim
from sklearn.metrics import accuracy_score, classification_report
from data_partition import clipDataTopX
import torch.nn as nn
from net.CNN import CNN_Model
from net.NN import NN_Model
from net.softMax import Softmax_Model
# from load_data import read_cifar10, read_cifar100, read_mnist
from dataset import readLocation_test,readLocation_train,readLocation_CTGAN_target,readPurchase50_train,readPurchase50_test,\
     readPurchase50_CTGAN_target,readPurchase100_train,readPurchase100_test,readPurchase100_CTGAN_target,\
     readPurchase10_test,readPurchase10_train,readPurchase10_CTGAN_target,readPurchase20_train,readPurchase20_test,\
     readPurchase20_CTGAN_target,readCIFAR10,readCIFAR100_StyleGAN_target,readCIFAR100_StyleGAN_shadow,readCIFAR100_StyleGAN_test,\
     readCIFAR100_StyleGAN_ori,readPurchase2_test,readPurchase2_train

from  collections import  Counter
from train import iterate_minibatches, CrossEntropy_L2
from data_partition import shuffleAndSplitData, clipDataTopX
from preprocessing import preprocessingCIFAR, preprocessingMINST, preprocessingNews, preprocessingAdult, \
    preprocessingCIFAR_GAN





def train_model(dataset, n_hidden=50, batch_size=100, epochs=100, learning_rate=0.01, model='cnn', l2_ratio=1e-7):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # train_x, train_y, totrain_x,totrain_y,test_x, test_y = dataset
    train_x, train_y,  test_x, test_y = dataset
    n_in = train_x.shape

    n_out = len(np.unique(train_y))

    if batch_size > len(train_y):
        batch_size = len(train_y)
    print('Building model with {} training data, {} classes...'.format(len(train_x), n_out))

    if model == 'cnn':
        print('Using a multilayer convolution neural network based model...')
        net = CNN_Model(n_in, n_hidden, n_out)
    elif model == 'nn':
        print('Using a multilayer neural network based model...')
        net = NN_Model(n_in, n_hidden, n_out)
    else:
        print('Using a single layer softmax based model...')
        net = Softmax_Model(n_in, n_out)

    # create loss function
    m = n_in[0]

    # criterion = CrossEntropy_L2(net, m, l2_ratio).to(device)
    # # criterion = nn.CrossEntropyLoss()
    #
    # net.to(device)
    #
    # # create optimizer
    # optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    #
    # # count loss in an epoch
    # temp_loss = 0.0
    #
    # # count the iteration number in an epoch
    # iteration = 0
    #
    # print('Training...')
    # net.train()
    # for epoch in range(epochs):
    #     for input_batch, target_batch in iterate_minibatches(train_x, train_y, batch_size):
    #         input_batch, target_batch = torch.tensor(input_batch), torch.tensor(target_batch).type(torch.long)
    #         input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    #         optimizer.zero_grad()
    #         outputs = net(input_batch)
    #         loss = criterion(outputs, target_batch)
    #         loss.backward()
    #         optimizer.step()
    #         temp_loss += loss.item()
    #
    #     temp_loss = round(temp_loss, 3)
    #     if epoch % 5 == 0:
    #         print('Epoch {}, train loss {}'.format(epoch, temp_loss))
    #
    #     temp_loss = 0.0
    #
    # net.eval()  # 把网络设定为训练状态

    #
    criterion = CrossEntropy_L2(net, m, l2_ratio).to(device)
    # criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.5)
    # optimizer = optim.Adam(net.parameters(), lr=learning_rate)

    visual_batch_size = 300
    visual_batch_rate = int(visual_batch_size / batch_size)

    privacy_engine = PrivacyEngine(
        net,
        batch_size=visual_batch_size,  # batch_size=256    epoch=90  lr=0.001
        sample_size=len(train_x),
        alphas=[1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64)),
        noise_multiplier=1.1,  # 1.1
        max_grad_norm=1.0,  # 1.0
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

        for i, (input_batch, target_batch) in enumerate(iterate_minibatches(train_x, train_y, batch_size)):

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
            if ((i + 1) % visual_batch_rate == 0) or ((i + 1) == int(len(train_x) / batch_size)):
                optimizer.step()
            else:
                optimizer.virtual_step()

            temp_loss += loss.item()

            if (i + 1) % 5 == 0:
                epsilon, best_alpha = optimizer.privacy_engine.get_privacy_spent(1e-5)
                print(
                    f"\tTrain Epoch: {epoch} \t"
                    f"Loss: {np.mean(losses):.6f} "
                    f"Acc@1: {np.mean(top1_acc):.6f} "
                    f"(ε = {epsilon:.2f}, δ = {1e-5}) for α = {best_alpha}"
                )

        temp_loss = 0.0

    net.eval()

    pred_y = []
    with torch.no_grad():
        for input_batch, _ in iterate_minibatches(train_x, train_y, batch_size, shuffle=False):
            input_batch = torch.tensor(input_batch)
            input_batch = input_batch.to(device)
            outputs = net(input_batch)
            pred_y.append(torch.max(outputs, 1)[1].data.cpu().numpy())
            # print(pred_y[:100])
        pred_y = np.concatenate(pred_y)
    print('Training Accuracy: {}'.format(accuracy_score(train_y, pred_y)))

    net.eval()  # 把网络设定为训练状态
    pred_y = []
    if test_x is not None:
        print('Testing...')
        if batch_size > len(test_y):
            batch_size = len(test_y)
        with torch.no_grad():
            for input_batch, _ in iterate_minibatches(test_x, test_y, batch_size, shuffle=False):
                input_batch = torch.tensor(input_batch)
                input_batch = input_batch.to(device)
                outputs = net(input_batch)
                pred_y.append(torch.max(outputs, 1)[1].data.cpu().numpy())
            pred_y = np.concatenate(pred_y)

        print('Testing Accuracy: {}'.format(accuracy_score(test_y, pred_y)))
    print('More detailed results:')
    print(classification_report(test_y, pred_y))

    return net


def train_target_model(dataset,
                       epochs=100,
                       batch_size=100,
                       learning_rate=0.01,
                       l2_ratio=1e-7,
                       n_hidden=50,
                       model='nn'):
    # train_x, train_y, totrain_x,totrain_y,test_x, test_y = dataset
    train_x, train_y,  test_x, test_y = dataset

    classifier_net = train_model(dataset=dataset,
                                 n_hidden=n_hidden,
                                 epochs=epochs,
                                 learning_rate=learning_rate,
                                 batch_size=batch_size,
                                 model=model,
                                 l2_ratio=l2_ratio)

    # test data for attack model
    attack_x, attack_y = [], []
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    classifier_net.eval()
    for batch, _ in iterate_minibatches(train_x, train_y, batch_size, False):
        batch = torch.tensor(batch)
        batch = batch.to(device)
        output = classifier_net(batch)
        preds_tensor = nn.functional.softmax(output, dim=1)
        attack_x.append(preds_tensor.detach().cpu().numpy())
        attack_y.append(np.ones(len(batch)))

    # data not used in training, label is 0
    for batch, _ in iterate_minibatches(test_x, test_y, batch_size, False):
        batch = torch.tensor(batch)
        # batch = Variable(batch)
        batch = batch.to(device)
        output = classifier_net(batch)
        preds_tensor = nn.functional.softmax(output, dim=1)
        attack_x.append(preds_tensor.detach().cpu().numpy())
        attack_y.append(np.zeros(len(batch)))

    attack_x = np.vstack(attack_x)
    attack_y = np.concatenate(attack_y)
    attack_x = attack_x.astype('float32')
    attack_y = attack_y.astype('int32')

    return attack_x, attack_y, classifier_net


def trainTarget(modelType,
                X, y,
                # train_target,
                # train_target_label,
                X_test=[], y_test=[],
                splitData=True,
                test_size=0.5,
                inepochs=50,
                batch_size=300,
                learning_rate=0.03):  #0.001
    if splitData:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    else:
        X_train = X
        y_train = y
        # train_target_x= np.array(train_target)
        # train_target_y = np.array(train_target_label)

    dataset = (X_train.astype(np.float32),
               y_train.astype(np.int32),
               # train_target_x.astype(np.float32),
               # train_target_y.astype(np.int32),
               X_test.astype(np.float32),
               y_test.astype(np.int32))

    attack_x, attack_y, theModel = train_target_model(dataset=dataset,
                                                      epochs=inepochs,
                                                      batch_size=batch_size,
                                                      learning_rate=learning_rate,
                                                      n_hidden=128,
                                                      l2_ratio=1e-07,
                                                      model=modelType)

    return attack_x, attack_y, theModel


def initializeTargetModel(dataset, num_epoch, classifierType='cnn'):
    # targetTrain, targetTrainLabel,train_target,train_target_label, targetTest, targetTestLabel = dataset
    targetTrain, targetTrainLabel,  targetTest, targetTestLabel = dataset
    attackModelDataTarget, attackModelLabelsTarget, targetModel = trainTarget(classifierType,
                                                                              targetTrain,
                                                                              targetTrainLabel,
                                                                              # train_target,
                                                                              # train_target_label,
                                                                              X_test=targetTest,
                                                                              y_test=targetTestLabel,
                                                                              splitData=False,
                                                                              inepochs=num_epoch,
                                                                              batch_size=300)

    return attackModelDataTarget, attackModelLabelsTarget, targetModel


def generateAttackData(dataset, classifierType, num_epoch, topX=3):
    targetX, targetY, target_model = initializeTargetModel(dataset, num_epoch, classifierType=classifierType)
    # 取目标模型和影子模型输出分布概率最高的前三个
    targetX = clipDataTopX(targetX, top=topX)

    return targetX, targetY, target_model


def top1_threshold_attack(x_, target_model):
    # 为每个数据集生成均匀分布的随机数据点
    # nonM_generated = np.random.uniform(0, 255, (1000, 3, 32, 32))  # mnist为:1*28*28  cifar10和cifar100都为:3*32*32
    # 然后将这些数据点输入到目标模型网络中获取每个数据点的最大后验概率值
    #生成的数据满足均匀分布
    # nonM_generated = np.random.randint(0,2,size=(1000,600))

    nonM_generated = np.random.choice([0, 1], size=(1000,600), p=[1.0/2, 1.0/2])

    result = [Counter(nonM_generated[i, :]).most_common(1)[0] for i in range(1000)]
    print("按行统计的结果为：", result[:50])

    # print(nonM_generated)
    # print(nonM_generated.shape)
    input = torch.tensor(nonM_generated).cuda().type(torch.float32)
    target_model.eval()
    with torch.no_grad():
        outputs = torch.nn.functional.softmax(target_model(input), dim=1)
        # print(outputs[:26])
        pre = torch.max(outputs, 1)[0].data.cpu().numpy()
        # print(pre[:100])
    # print(pre.shape)
    # 通过设置百分位数t,来计算阈值的公式:b[(len(b) - 1 ) * q % + 1]
    threshold = np.percentile(pre, 20, interpolation='lower')  # linear, lower, higher, midpoint, nearest
    # threshold =0.9972524
    print('threshold=', threshold)
    # 如果最大后验概率值大于阈值设置为1否则设置为0
    m_pred = np.where(x_.max(axis=1) > threshold, 1, 0)

    return m_pred



def shuffleAndSplitData_(dataX, dataY, cluster):
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


    return toTrainData, toTrainLabel, toTestData, toTestLabel



if __name__ == '__main__':
    # originalDatasetPath = './data_row/cifar10/cifar-10-batches-py'  # ../data_row/cifar-100-python  #../data_row/MNIST

    toTrainData, toTrainLabel,_,_ = readPurchase100_train()
    # shadowdata,shadowlabel = readCIFAR100_StyleGAN_shadow()
    # trainData,trainlabel,  _, _ = readCIFAR100_StyleGAN_ori()
    # datax,datay = readCIFAR100_StyleGAN_test()
    toTestData,toTestLabel,_,_ = readPurchase100_test()
    # cluster = 10520
    # toTestData,toTestLabel,_,_ = shuffleAndSplitData_(datax,datay,cluster)
    # toTrainData,shadowdata = preprocessingCIFAR_GAN(toTrainData,shadowdata)
    # trainData,toTestData = preprocessingCIFAR_GAN(trainData,toTestData)
    # result = [Counter(toTrainData[i, :]).most_common(1)[0] for i in range(1000)]
    # print("训练数据按行统计的结果为：", result[:50])

    # cluster =25000
    # toTrainData, toTrainLabel ,toTestData, toTestLabel = shuffleAndSplitData_(toTrainData,toTrainLabel,cluster)
    # toTrainData,toTestData = preprocessingCIFAR(toTrainData,toTestData)

    # toTrainData = np.concatenate((toTrainData,shadowdata),axis=0)
    # toTrainLabel = np.concatenate((toTrainLabel,shadowlabel_),axis=0)


    # result = [Counter(toTestData[i, :]).most_common(1)[0] for i in range(1000)]
    # print("测试数据按行统计的结果为：", result[:50])
    # trainData , trainDatalabel = readPurchase20_CTGAN_target()
    print(toTrainData.shape)
    print(toTestData.shape)
    # print(trainData.shape)

    dataset = (toTrainData.astype(np.float32),
               toTrainLabel.astype(np.int32),
               # trainData.astype(np.float32),
               # trainlabel.astype(np.int32),
               toTestData.astype(np.float32),
               toTestLabel.astype(np.int32))

    # 训练目标模型，并获取真实的0/1标签
    attack_x, attack_y, model = generateAttackData(dataset, classifierType='nn', num_epoch=400, topX=3)
    # print(attack_y)
    # 通过设置的阈值得到的0/1标签数组
    m_pred = top1_threshold_attack(attack_x, target_model=model)
    # # 通过比较真实标签和预测的标签来显示预测值
    print('chose Threshold train Accuracy: {}'.format(accuracy_score(attack_y, m_pred)))
    mc = classification_report(attack_y, m_pred)
    print(mc)









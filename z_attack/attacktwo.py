import numpy as np

from data_partition import clipDataTopX
from initialization import load_data
from net.softMax import Softmax_Model
from paint import Paint
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import multilabel_confusion_matrix

import warnings

from train import CrossEntropy_L2, iterate_minibatches, trainAttackModel

warnings.filterwarnings("ignore")

seed = 21312
np.random.seed(seed)  # Numpy module.


def train_model(dataset, n_hidden=50, batch_size=100, epochs=100, learning_rate=0.01, l2_ratio=1e-7):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_x, train_y, test_x, test_y = dataset
    n_in = train_x.shape
    n_out = len(np.unique(train_y))

    if batch_size > len(train_y):
        batch_size = len(train_y)
    print('Building model with {} training data, {} classes...'.format(len(train_x), n_out))

    net = Softmax_Model(n_in, n_hidden, n_out)

    # create loss function
    m = n_in[0]
    # create optimizer
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    criterion = CrossEntropy_L2(net, m, l2_ratio).to(device)

    net.to(device)

    for epoch in range(epochs):
        net = train(train_x, train_y, batch_size, net, epoch, optimizer, criterion)
        test(train_x, train_y, batch_size, net, "Training_accuary")
        if test_x is not None:
            if batch_size > len(test_y):
                batch_size = len(test_y)
            pred_y = test(test_x, test_y, batch_size, net, "Testing_accuary")

    print('More detailed results:')
    print(classification_report(test_y, pred_y))

    mcm = multilabel_confusion_matrix(test_y, pred_y, labels=[0, 1])
    print(mcm)

    return net


def train(train_x, train_y, batch_size, net, epoch, optimizer, criterion):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.train()
    temp_loss = 0.0
    for input_batch, target_batch in iterate_minibatches(train_x, train_y, batch_size):
        input_batch, target_batch = torch.tensor(input_batch), torch.tensor(target_batch).type(torch.long)
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
    # if (epoch % 5 == 0):
    print('Epoch {}, train loss {}'.format(epoch, temp_loss))

    return net


def test(test_x, test_y, batch_size, net, Flag):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.eval()
    pred_y = []

    with torch.no_grad():
        for input_batch, _ in iterate_minibatches(test_x, test_y, batch_size, shuffle=False):
            input_batch = torch.tensor(input_batch)
            input_batch = input_batch.to(device)

            outputs = net(input_batch)

            pred_y.append(torch.max(outputs, 1)[1].data.cpu().numpy())

        pred_y = np.concatenate(pred_y)

    if (Flag == "Training_accuary"):
        print('Training_accuary: {}'.format(accuracy_score(test_y, pred_y)))
    else:
        print('Testing_accuary: {}'.format(accuracy_score(test_y, pred_y)))

    return pred_y


# ./Adult/5600/targetModelData.npz
# ./CIFAR100/10520/targetModelData.npz
# ./Face/420/targetModelData.npz
# ./MINST/10520/targetModelData.npz
# ./News/4500/targetModelData.npzLL
shadowX, shadowY = load_data("../dataCIFAR10/10520/shadowModelData.npz")
targetX, targetY = load_data("../dataCIFAR10/10520/targetModelData.npz")
#
# shadowX, shadowY = load_data("../dataPurchase/purchase2/10005/shadowModelData.npz")
# targetX, targetY = load_data("../dataPurchase/purchase2/10005/targetModelData.npz")

targetX = clipDataTopX(targetX, top=3)
shadowX = clipDataTopX(shadowX, top=3)

# 训练攻击模型
net = trainAttackModel(shadowX, shadowY, targetX, targetY,model='normal')

modelPath = "../model/CIFAR10"
torch.save(net, modelPath + '/net3.pkl')

batch_size = 10


# 用攻击模型去攻击其他数据
print("============================================================================")
# print("攻击cifar10-DPSGD")
# targetX, targetY = load_data("../dataLocation/1250/targetModelData.npz")
targetX, targetY = load_data("../dataDPSGD/dataMINST/200_100_0.01/targetModel.npz")
# targetX, targetY = load_data('../dataPurchase/purchase100/10005_kmeans/targetModelData.npz')
# targetX, targetY = load_data("../dataPurchase/purchase10/10005_kmeans/targetModelData.npz")
# targetX, targetY = load_data("../dataPurchase/purchase50/10005_epoch200_lr0.001/targetModelData.npz")


targetX = clipDataTopX(targetX, top=3)
shadowX = clipDataTopX(shadowX, top=3)

pred_y = test(targetX, targetY, batch_size, net, "Testing_accuary")
print('More detailed results:')
print(classification_report(targetY, pred_y))
mcm = multilabel_confusion_matrix(targetY, pred_y, labels=[0, 1])
print(mcm)


#
# print("============================================================================")
# print("攻击Cifar10")
# targetX, targetY = load_data("./dataCIFAR10/10520/targetModelData.npz")
# targetX = clipDataTopX(targetX, top=2)
# pred_y = test(targetX, targetY, batch_size, net, "Testing_accuary")
# print('More detailed results:')
# print(classification_report(targetY, pred_y))
# mcm = multilabel_confusion_matrix(targetY, pred_y, labels=[0, 1])
# print(mcm)
#
# print("============================================================================")
# print("攻击Cifar100")
# targetX, targetY = load_data("./dataCIFAR100/10520/targetModelData.npz")
# targetX = clipDataTopX(targetX, top=2)
# pred_y = test(targetX, targetY, batch_size, net, "Testing_accuary")
# print('More detailed results:')
# print(classification_report(targetY, pred_y))
# mcm = multilabel_confusion_matrix(targetY, pred_y, labels=[0, 1])
# print(mcm)
#
# print("============================================================================")
# print("攻击Face")
# targetX, targetY = load_data("./dataLFW/420/targetModelData.npz")
# targetX = clipDataTopX(targetX, top=2)
# pred_y = test(targetX, targetY, batch_size, net, "Testing_accuary")
# print('More detailed results:')
# print(classification_report(targetY, pred_y))
# mcm = multilabel_confusion_matrix(targetY, pred_y, labels=[0, 1])
# print(mcm)
#
# print("============================================================================")
# print("攻击Mnist")
# targetX, targetY = load_data("./dataMINST/10520/targetModelData.npz")
# targetX = clipDataTopX(targetX, top=2)
# pred_y = test(targetX, targetY, batch_size, net, "Testing_accuary")
# print('More detailed results:')
# print(classification_report(targetY, pred_y))
# mcm = multilabel_confusion_matrix(targetY, pred_y, labels=[0, 1])
# print(mcm)
#
# print("============================================================================")
# print("攻击News")
# targetX, targetY = load_data("./dataNews/4500/targetModelData.npz")
# targetX = clipDataTopX(targetX, top=2)
# pred_y = test(targetX, targetY, batch_size, net, "Testing_accuary")
# print('More detailed results:')
# print(classification_report(targetY, pred_y))
# mcm = multilabel_confusion_matrix(targetY, pred_y, labels=[0, 1])
# print(mcm)

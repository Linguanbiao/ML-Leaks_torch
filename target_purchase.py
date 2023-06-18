import  torch
import  torch.nn as  nn
import numpy as  np
import random

# from numpy.core import moveaxis
from sklearn.metrics import accuracy_score

from data_partition import clipDataTopX
from  dataset  import  readcifar10_GAN,readCIFAR10,readcifar10_target_train,readcifar10_StyleGAN_all,\
      readcifar10_StyleGAN_test
from net.CNN import CNN_Model
from net.NN import NN_Model
from preprocessing import preprocessingCIFAR,preprocessingCIFAR_GAN
from train import CrossEntropy_L2, iterate_minibatches
from torch import optim
import  pandas as  pd

def shuffleAndSplitData(dataX, dataY, cluster):

    # c = list(zip(dataX, dataY))
    # random.shuffle(c)
    # dataX, dataY = zip(*c)

    data_x = np.array(dataX[:cluster])
    data_y = np.array(dataY[:cluster])

    return  data_x,data_y


def train_traget_model(toTrainData,
                       toTrainLabel,
                       gantestdata,
                       gantestlabel,
                       orTrainData,
                       orTrainLabel,
                       toTestData_,
                       toTestLabel_,
                       epochs=50,
                       batch_size=100,
                       model='nn',
                       n_hidden=128,
                       l2_ratio=1e-07,
                       learning_rate=0.001):

    toTrainData = toTrainData.astype(np.float32)
    toTrainLabel = toTrainLabel.astype(np.int32)
    gantestdata =gantestdata.astype(np.float32)
    gantestlabel =gantestlabel.astype(np.int32)
    orTrainData = orTrainData.astype(np.float32)
    orTrainLabel = orTrainLabel.astype(np.int32)
    toTestData = toTestData_.astype(np.float32)
    toTestLabel = toTestLabel_.astype(np.int32)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    n_in = toTrainData.shape
    n_out = len(np.unique(toTrainLabel))

    if batch_size > len(toTrainLabel):
        batch_size = len(toTrainLabel)

    net = NN_Model(n_in, n_hidden, n_out)
    net.to(device)

    m = n_in[0]
    criterion = CrossEntropy_L2(net, m, l2_ratio).to(device)

    optimizer = optim.Adam(net.parameters(), lr=learning_rate)

    print('Training...')
    net.train()

    temp_loss = 0.0

    for epoch in range(epochs):

        for input_batch, target_batch in iterate_minibatches(toTrainData, toTrainLabel, batch_size):
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

        for input_batch, _ in iterate_minibatches(gantestdata,gantestlabel, batch_size, shuffle=False):
            input_batch = torch.tensor(input_batch).contiguous()
            input_batch = input_batch.to(device)

            outputs = net(input_batch)

            pred_y.append(torch.max(outputs, 1)[1].data.cpu().numpy())

        pred_y = np.concatenate(pred_y)

    print('gan test train Accuracy: {}'.format(accuracy_score(gantestlabel, pred_y)))

    pred_y = []
    confidencex=[]
    confidencey=[]
    with torch.no_grad():

        for input_batch, _ in iterate_minibatches(orTrainData, orTrainLabel, batch_size, shuffle=False):
            input_batch = torch.tensor(input_batch).contiguous()
            input_batch = input_batch.to(device)

            outputs = net(input_batch)
            confidence = nn.functional.softmax(outputs)
            confidence = clipDataTopX(confidence.detach().cpu().numpy())
            confidencex.append(confidence)
            confidencey.append(np.ones(len(input_batch)))

            pred_y.append(torch.max(outputs, 1)[1].data.cpu().numpy())

        pred_y = np.concatenate(pred_y)
        confidencex=np.concatenate(confidencex,axis=0)
        confidencey=np.concatenate(confidencey,axis=0)

    print('ori Train Accuracy: {}'.format(accuracy_score(orTrainLabel, pred_y)))

    # 测试
    pred_y = []
    test_confidencex=[]
    test_confidencey=[]

    with torch.no_grad():

        for input_batch, _ in iterate_minibatches(toTestData, toTestLabel, batch_size, shuffle=False):
            input_batch = torch.tensor(input_batch).contiguous()
            input_batch = input_batch.to(device)

            outputs = net(input_batch)
            confidence = nn.functional.softmax(outputs)
            confidence = clipDataTopX(confidence.detach().cpu().numpy())
            test_confidencex.append(confidence)
            test_confidencey.append(np.zeros(len(input_batch)))

            pred_y.append(torch.max(outputs, 1)[1].data.cpu().numpy())

        pred_y = np.concatenate(pred_y)
        test_confidencex=np.concatenate(test_confidencex,axis=0)
        test_confidencey=np.concatenate(test_confidencey,axis=0)

    print('ori Test Accuracy: {}'.format(accuracy_score(toTestLabel, pred_y)))

    confidencex=np.concatenate((confidencex,test_confidencex),axis=0)
    confidencey=np.concatenate((confidencey,test_confidencey),axis=0)
    data_pathtosave = 'dataCTGAN/dataPurchase50/10005'
    np.savez(data_pathtosave+'/targetModel.npz',confidencex,confidencey)

def readPurchase50_train():
    # print("use max_label")
    data_path = 'data/purchase/purchase50/GAN_custom_kmeans_targetmodel_max_10005 .csv'
    f = pd.read_csv(data_path, header=None, skiprows=1,nrows=30000)

    # normalize the values
    x_range = [i for i in range(600)]
    trainX = f[x_range].values
    trainY = f[600].values

    return trainX, trainY

def readPurchase50_gan_test():
    # print("use max_label")
    data_path = 'data/purchase/purchase50/GAN_custom_kmeans_targetmodel_max_10005 .csv'
    f = pd.read_csv(data_path, header=None, skiprows=10005,nrows=10005)

    # normalize the values
    x_range = [i for i in range(600)]
    trainX = f[x_range].values
    trainY = f[600].values

    return trainX, trainY

def readPurchase50_ori_train():
    # print("use max_label")
    data_path = 'data/purchase/purchase50/max_label.csv'
    f = pd.read_csv(data_path, header=None, skiprows=1,nrows=10005)

    # normalize the values
    x_range = [i for i in range(600)]
    trainX = f[x_range].values
    trainY = f[600].values

    return trainX, trainY

def readPurchase50_test():
    # print("use max_label")
    data_path = 'data/purchase/purchase50/max_label.csv'
    f = pd.read_csv(data_path, header=None, skiprows=10005,nrows=10005)

    # normalize the values
    x_range = [i for i in range(600)]
    trainX = f[x_range].values
    trainY = f[600].values

    return trainX, trainY

if __name__ == '__main__':

    toTrainData_, toTrainLabel_ = readPurchase50_train()
    gantestdata,gantestlabel = readPurchase50_gan_test()
    orTrainData_,orTrainLabel_ = readPurchase50_ori_train()
    toTestData_,toTestLabel_ = readPurchase50_test()

    # cluster=10005
    # toTestData_,toTestLabel_ = shuffleAndSplitData(toTestData_,toTestLabel_,cluster)

    print(toTrainData_.shape)
    print(toTestData_.shape)
    print(orTrainData_.shape)
    print(gantestdata.shape)
    # print(toTrainLabel_)
    # print(toTestLabel_)

    train_traget_model(toTrainData=toTrainData_,
                       toTrainLabel=toTrainLabel_,
                       gantestdata =gantestdata,
                       gantestlabel =gantestlabel,
                       orTrainData= orTrainData_,
                       orTrainLabel= orTrainLabel_,
                       toTestData_ = toTestData_,
                       toTestLabel_ = toTestLabel_,
                       epochs=200,
                       batch_size=100,
                       model='cnn',
                       n_hidden=128,
                       l2_ratio=1e-07,
                       learning_rate=0.001)

    # toTestData=or_testdata,
    # toTestLabel=or_testlabel,
    # or_traindata=or_traindata,
    # or_trainlabel=or_trainlabel,




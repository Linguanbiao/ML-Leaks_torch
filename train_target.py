import  torch
import  torch.nn as  nn
import numpy as  np
import random

from sklearn.metrics import accuracy_score

from data_partition import clipDataTopX
from  dataset  import  readcifar10_GAN,readCIFAR10,readAdult_fake,readAdult_test,readAdult_train,readNews
from net.NN import NN_Model
from preprocessing import preprocessingCIFAR,preprocessingCIFAR_GAN,preprocessingAdult,preprocessingNews
from train import CrossEntropy_L2, iterate_minibatches
from torch import optim

def shuffleAndSplitData(dataX, dataY, cluster):

    c = list(zip(dataX, dataY))
    random.shuffle(c)
    dataX, dataY = zip(*c)

    toTrainData = np.array(dataX[:cluster])
    toTrainLabel = np.array(dataY[:cluster])

    toTestData = np.array(dataX[cluster:cluster * 2])
    toTestLabel = np.array(dataY[cluster:cluster * 2])


    return toTrainData, toTrainLabel, toTestData, toTestLabel


def train_traget_model(toTrainData,
                       toTrainLabel,
                       toTestData,
                       toTestLabel,
                       epochs=50,
                       batch_size=100,
                       model='cnn',
                       n_hidden=128,
                       l2_ratio=1e-07,
                       learning_rate=0.001):

    toTrainData = toTrainData.astype(np.float32)
    toTrainLabel = toTrainLabel.astype(np.int32)
    toTestData = toTestData.astype(np.float32)
    toTestLabel = toTestLabel.astype(np.int32)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    n_in = toTrainData.shape

    n_out = len(np.unique(toTrainLabel))

    if batch_size > len(toTrainLabel):
        batch_size = len(toTrainLabel)

    # net = CNN_Model(n_in, n_hidden, n_out)
    net = NN_Model(n_in,n_hidden,n_out)
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

    # pred_y = []
    # confidencex=[]
    # confidencey=[]
    # with torch.no_grad():
    #
    #     for input_batch, _ in iterate_minibatches(or_traindata, or_trainlabel, batch_size, shuffle=False):
    #         input_batch = torch.tensor(input_batch).contiguous()
    #         input_batch = input_batch.to(device)
    #
    #         outputs = net(input_batch)
    #         confidence = nn.functional.softmax(outputs)
    #         confidence = clipDataTopX(confidence.detach().cpu().numpy())
    #         confidencex.append(confidence)
    #         confidencey.append(np.ones(len(input_batch)))
    #
    #         pred_y.append(torch.max(outputs, 1)[1].data.cpu().numpy())
    #
    #     pred_y = np.concatenate(pred_y)
    #     confidencex=np.concatenate(confidencex,axis=0)
    #     confidencey=np.concatenate(confidencey,axis=0)

    # print('original Train Accuracy: {}'.format(accuracy_score(or_trainlabel, pred_y)))

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

    print('Test Accuracy: {}'.format(accuracy_score(toTestLabel, pred_y)))
    #
    confidencex=np.concatenate((confidencex,test_confidencex),axis=0)
    confidencey=np.concatenate((confidencey,test_confidencey),axis=0)
    data_pathtosave = '../dataCTGAN/dataAdult/5600'
    np.savez(data_pathtosave+'/targetModel.npz',confidencex,confidencey)


if __name__ == '__main__':
    originalDatasetPath = '../data/News'
    dataX,dataY = readNews (originalDatasetPath)
    cluster = 4500
    toTrainData,toTrainLabel,toTestData,toTestLabel = shuffleAndSplitData(dataX,dataY,cluster)

    or_traindata,or_testdata = preprocessingNews(toTrainData,toTestData)

    print(or_traindata.shape)
    print(or_testdata.shape)
    print(or_traindata[:10])
    print(or_testdata[:10])


    train_traget_model(toTrainData=or_traindata,
                       toTrainLabel=toTrainLabel,
                       toTestData=or_testdata,
                       toTestLabel=toTestLabel,
                       epochs=50,
                       batch_size=100,
                       model='cnn',
                       n_hidden=128,
                       l2_ratio=1e-07,
                       learning_rate=0.001)




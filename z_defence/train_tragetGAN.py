import  torch
import  torch.nn as  nn
import numpy as  np
import random

from sklearn.metrics import accuracy_score

from data_partition import clipDataTopX
from  dataset  import  readcifar10_GAN,readCIFAR10,readAdult_fake,readAdult_test,readAdult_train,readNews,\
    readPurchase100_CTGAN_target,readPurchase100_test,readPurchase100_train,readLocation_CTGAN_target,readLocation_train,\
    readLocation_test,readPurchase10_CTGAN_target,readPurchase10_test,readPurchase10_train,readMINTS_StyleGAN_ori,\
    readMINTS_StyleGAN_test,readMINST_StyleGAN_target,readMINST_StyleGAN_shadow,readPurchase20_CTGAN_target,readPurchase20_train,\
    readPurchase20_test,readPurchase2_test,readPurchase2_train,readPurchase2_CTGAN_target,readCIFAR100_StyleGAN_ori,readCIFAR100_StyleGAN_test,\
    readCIFAR100_StyleGAN_target,readCIFAR100_StyleGAN_shadow,readCIFAR100
from net.NN import NN_Model
from  net.CNN import CNN_Model
from preprocessing import preprocessingCIFAR,preprocessingCIFAR_GAN,preprocessingAdult,preprocessingNews,preprocessingMINST
from train import CrossEntropy_L2, iterate_minibatches
from torch import optim

def shuffleAndSplitData(dataX, dataY, cluster):

    c = list(zip(dataX, dataY))
    random.shuffle(c)
    dataX, dataY = zip(*c)

    data_x = np.array(dataX[:cluster])
    data_y = np.array(dataY[:cluster])
    datax  = np.array(dataX[cluster:cluster*2])
    datay = np.array(dataY[cluster:cluster*2])

    return  data_x,data_y,datax,datay


def train_traget_model(toTrainData,
                       toTrainLabel,
                       toTestData,
                       toTestLabel,
                       # or_traindata,
                       # or_trainlabel,
                       epochs=50,
                       batch_size=100,
                       model='cnn',
                       n_hidden=128,
                       l2_ratio=1e-07,
                       learning_rate=0.001):

    # toTrainData = toTrainData.astype(np.float32)
    # # toTrainLabel = toTrainLabel.astype(np.int32)
    # toTestData = toTestData.astype(np.float32)
    # toTestLabel = toTestLabel.astype(np.int32)
    # or_traindata = or_traindata.astype(np.float32)
    # or_trainlabel = or_trainlabel.astype(np.int32)

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

    # optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.5)

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
    #
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

    # confidencex=np.concatenate((confidencex,test_confidencex),axis=0)
    # confidencey=np.concatenate((confidencey,test_confidencey),axis=0)
    # data_pathtosave = '../dataCTGAN/dataPurchase2/10005'
    # np.savez(data_pathtosave+'/targetModel.npz',confidencex,confidencey)


if __name__ == '__main__':
    # originalDatasetPath = '../data/News'
    toTrainData_, toTrainLabel_ , testx, testy = readCIFAR100()

    # shadowdata,shadowtest = readCIFAR100_StyleGAN_shadow()
    toTrainData_ = toTrainData_.astype(float)


    toTrainData_ ,testx = preprocessingCIFAR(toTrainData_,testx)

    # data_path = '../data/Adult'
    # or_traindata_,or_trainlabel_,or_testdata_,or_testlabel_ = readCIFAR10(data_path)

    # or_traindata_, or_trainlabel_, _, _ = readCIFAR100_StyleGAN_ori()
    # datax,datay = readLocation_test()
    # cluster=1250
    # datax,datay = readCIFAR100_StyleGAN_test()
    # cluster = 10520
    # #
    # or_testdata_,or_testlabel,_,_ = shuffleAndSplitData(datax,datay,cluster)
    #
    # toTrainData_ ,_= preprocessingMINST(toTrainData_,shadowdata)
    # or_traindata_,or_testdata_ = preprocessingCIFAR_GAN(or_traindata_,or_testdata_)
    # toTrainData_,toshadowdata = preprocessingCIFAR_GAN(toTrainData_,shadowdata)
    # or_traindata,or_trainlabel = shuffleAndSplitData(or_traindata_,or_trainlabel_,cluster)
    # or_testdata, or_testlabel = shuffleAndSplitData(or_testdata_, or_testlabel_, cluster)
    #
    # toTrainData, or_traindata,or_testdata = preprocessingAdult(toTrainData,or_traindata,or_testdata)
    # # toTrainData = preprocessingCIFAR_GAN(toTrainData)

    train_traget_model(toTrainData=toTrainData_,
                       toTrainLabel=toTrainLabel_,
                       # toTestData=or_testdata_,
                       # toTestLabel=or_testlabel,

                       toTestData=testx,
                       toTestLabel=testy,
                       # or_traindata=or_traindata_,
                       # or_trainlabel=or_trainlabel_,
                       epochs=50,
                       batch_size=250,
                       model='cnn',
                       n_hidden=128,
                       l2_ratio=1e-07,
                       learning_rate=0.1)




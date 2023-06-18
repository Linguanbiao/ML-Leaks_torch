import  torch
import  torch.nn as  nn
import numpy as  np
import random

from opacus import PrivacyEngine
from opacus.utils import module_modification
from sklearn.metrics import accuracy_score

from data_partition import clipDataTopX,shuffleAndSplitData
from dataset import readcifar10_GAN, readCIFAR10, readCIFAR100,readMINST,readNews,readFLW,readLocation_train,readLocation_test,\
    readPurchase10_test,readPurchase10_train,readPurchase50_test,readPurchase50_train,readPurchase2_train,\
    readPurchase2_test,readPurchase20_train,readPurchase20_test,readPurchase100_test,readPurchase100_train


from net.CNN import CNN_Model
from net.NN import NN_Model
from paint_distribution import paint_histogram
from preprocessing import preprocessingCIFAR,preprocessingCIFAR_GAN, preprocessingMINST,preprocessingNews,preprocessingLFW
from train import CrossEntropy_L2, iterate_minibatches
from torch import optim

# def shuffleAndSplitData(dataX, dataY, cluster):
#
#     c = list(zip(dataX, dataY))
#     random.shuffle(c)
#     dataX, dataY = zip(*c)
#
#     data_x = np.array(dataX[:cluster])
#     data_y = np.array(dataY[:cluster])
#
#     return  data_x,data_y

def shuffleAndSplitData_news(dataX, dataY, cluster):

    c = list(zip(dataX, dataY))
    random.shuffle(c)
    dataX, dataY = zip(*c)

    data_x = np.array(dataX[:cluster])
    data_y = np.array(dataY[:cluster])
    test_x = np.array(dataX[cluster:cluster*2])
    test_y = np.array(dataY[cluster:cluster*2])

    return  data_x,data_y,test_x,test_y



def train_traget_model(toTrainData,
                       toTrainLabel,
                       toTestData,
                       toTestLabel,
                       epochs=50,
                       batch_size=100,
                       model='cnn',
                       n_hidden=128,
                       l2_ratio=1e-07,
                       learning_rate=0.02
                       ):

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
    # net = module_modification.convert_batchnorm_modules(net)

    # net = NN_Model(n_in, n_hidden, n_out)
    net.to(device)
    m = n_in[0]

    criterion = CrossEntropy_L2(net, m, l2_ratio).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=learning_rate,momentum=0.5)

    # optimizer_ExpLR = torch.optim.SGD(net.parameters(), lr=0.01)
    # ExpLR = torch.optim.lr_scheduler.StepLR(optimizer_ExpLR, step_size=5, gamma=0.9)

    # optimizer = optim.Adam(net.parameters(), lr=learning_rate)

    visual_batch_size = 100
    visual_batch_rate = int(visual_batch_size / batch_size)

    privacy_engine = PrivacyEngine(
        net,
        batch_size=visual_batch_size,  # batch_size=256    epoch=90  lr=0.001
        sample_size=len(toTrainLabel),
        alphas=[1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64)),
        noise_multiplier=1.1,  # 1.1
        max_grad_norm =1.0,  # 1.0
        secure_rng=False,
        target_delta=1e-5
    )
    privacy_engine.attach(optimizer)
    # privacy_engine.attach(optimizer_ExpLR)
    # privacy_engine.attach(ExpLR)


    print('Training...')
    net.train()

    temp_loss = 0.0
    top1_acc = []
    losses = []

    for epoch in range(epochs):

        for i, (input_batch, target_batch) in enumerate(iterate_minibatches(toTrainData, toTrainLabel, batch_size)):

            input_batch, target_batch = torch.tensor(input_batch).contiguous(), torch.tensor(target_batch).type(torch.long)
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
                # optimizer_ExpLR.step()
            else:
                # optimizer_ExpLR.step()
                optimizer.step()
                # ExpLR.virtual_step()

            temp_loss += loss.item()

            if (i + 1) % 5 == 0:
                epsilon, best_alpha = optimizer.privacy_engine.get_privacy_spent(1e-5)
                print(
                    # f"\tLearning rate: {ExpLR.get_lr()} \t"
                    f"Train Epoch: {epoch} "
                    f"Loss: {np.mean(losses):.6f} "
                    f"Acc@1: {np.mean(top1_acc):.6f} "
                    f"(ε = {epsilon:.4f}, δ = {1e-5}) for α = {best_alpha}"
                )
        # ExpLR.step()
        temp_loss = 0.0


    net.eval()

    #训练集精度
    pred_y = []
    confidencex=[]
    confidencey=[]
    with torch.no_grad():

        for input_batch, _ in iterate_minibatches(toTrainData, toTrainLabel, batch_size, shuffle=False):
            input_batch = torch.tensor(input_batch).contiguous()
            input_batch = input_batch.to(device)

            outputs = net(input_batch)
            confidence = nn.functional.softmax(outputs, dim=1)
            confidence = clipDataTopX(confidence.detach().cpu().numpy())
            confidencex.append(confidence)
            confidencey.append(np.ones(len(input_batch)))

            pred_y.append(torch.max(outputs, 1)[1].data.cpu().numpy())

        pred_y = np.concatenate(pred_y)
        confidencex=np.concatenate(confidencex,axis=0)
        confidencey=np.concatenate(confidencey,axis=0)

    print('Train Accuracy: {}'.format(accuracy_score(toTrainLabel, pred_y)))

    # 测试
    pred_y = []
    test_confidencex=[]
    test_confidencey=[]

    with torch.no_grad():

        for input_batch, _ in iterate_minibatches(toTestData, toTestLabel, batch_size, shuffle=False):
            input_batch = torch.tensor(input_batch).contiguous()
            input_batch = input_batch.to(device)

            outputs = net(input_batch)
            confidence = nn.functional.softmax(outputs, dim=1)
            confidence = clipDataTopX(confidence.detach().cpu().numpy())
            test_confidencex.append(confidence)
            test_confidencey.append(np.zeros(len(input_batch)))

            pred_y.append(torch.max(outputs, 1)[1].data.cpu().numpy())

        pred_y = np.concatenate(pred_y)
        test_confidencex=np.concatenate(test_confidencex,axis=0)
        test_confidencey=np.concatenate(test_confidencey,axis=0)



    print('Test Accuracy: {}'.format(accuracy_score(toTestLabel, pred_y)))

    confidencex_=np.concatenate((confidencex,test_confidencex),axis=0)
    confidencey_=np.concatenate((confidencey,test_confidencey),axis=0)
    # data_pathtosave = '../dataDPSGD/dataPurchase50/test/p10_50_50_0.01_sgd'
    # np.savez(data_pathtosave+'/targetModel.npz',confidencex_,confidencey_)
    # paint_histogram(confidencex, test_confidencex)

def shuffleData(datax, datay):
    c = list(zip(datax, datay))
    random.shuffle(c)
    dataX, dataY = zip(*c)
    dataX = np.array(dataX)
    dataY = np.array(dataY)

    return dataX,dataY

if __name__ == '__main__':

    # data_path = '../data/MINST'
    # # traindata,trainlabel,testdata,testlabel= readCIFAR100(data_path)
    # targetx,targety,_ ,_  = readPurchase100_train()
    # target_testx,target_testy,_ ,_ = readPurchase100_test()
    # dataX, dataY ,testx,testy= readCIFAR10()
    # cluster
    # targetx,targety,target_testx,target_testy,_,_,_,_ = shuffleAndSplitData(dataX,dataY,cluster)
    targetx, targety, target_testx, target_testy = readMINST()
    targetx,target_testx= preprocessingMINST(targetx,target_testx)

    print(targetx.shape)
    print(target_testx.shape)
    # # cluster = 4500
    #
    # # traindata,trainlabel = shuffleAndSplitData(traindata,trainlabel,cluster)
    # # traindata, trainlabel,testdata,testlabel = shuffleAndSplitData_news(datax, datay, cluster)
    #
    # # traindata,testdata = preprocessingCIFAR(traindata,testdata)
    # # traindata,testdata = preprocessingNews(traindata,testdata)


    # datax,datay,_,_ = readMINST(data_path)
    # print(datax.shape)
    # cluster = 10520
    # toTrainData, toTrainLabel, shadowData, shadowLabel, toTestData, toTestLabel, shadowTestData, shadowTestLabel = \
    #     shuffleAndSplitData(datax, datay, cluster)
    # targetx,target_testx = preprocessingMINST(toTrainData,toTestData)



    train_traget_model(toTrainData=targetx,
                       toTrainLabel=targety,
                       # toTrainLabel=toTrainLabel,
                       toTestData=target_testx,
                       toTestLabel=target_testy,
                       # toTestLabel=toTestLabel,
                       epochs=10,
                       batch_size=100,
                       model='nn',
                       n_hidden=128,
                       l2_ratio=1e-07,
                       learning_rate=0.03
                       )




# ！-*- coding:utf-8 -*-
import os
import random

import numpy as np
import torch

from data_partition import shuffleAndSplitData, clipDataTopX
from dataset import readCIFAR10, readCIFAR100, readMINST, readNews, readFLW, readAdult, readcifar10_GAN, \
    readcifar10_StyleGAN_all, readcifar10_StyleGAN_shadow, readcifar10_StyleGAN_target, readcifar10_shadow_test, \
    readcifar10_target_test,readPurchase50_train,readPurchase50_CTGAN_shadow,readPurchase50_CTGAN_target,\
    readLocation,readPurchase50,readPurchase100,readPurchase100_CTGAN_shadow,readPurchase100_CTGAN_target,\
    readPurchase100_test,readPurchase100_train,readLocation_train,readLocation_test,readLocation_CTGAN_target,\
    readLocation_CTGAN_shadow,readPurchase10,readPurchase10_train,readPurchase10_test,readPurchase10_CTGAN_shadow,\
    readPurchase10_CTGAN_target,readMINST_StyleGAN_shadow,readMINST_StyleGAN_target,readMINTS_StyleGAN_ori,\
    readMINTS_StyleGAN_test,readPurchase20_train,readPurchase20_test,readPurchase20_CTGAN_shadow,readPurchase20_CTGAN_target,\
    readPurchase20,readMNISTBin,readCIFAR100_StyleGAN_shadow,readCIFAR100_StyleGAN_target,readCIFAR100_StyleGAN_ori,\
    readCIFAR100_StyleGAN_test,readPurchase2_CTGAN_shadow,readPurchase2_CTGAN_target,readPurchase2_test,readPurchase2_train,\
    readPurchase2

from dataset import readcifar10_shadow_train, readcifar10_target_train,readPurchase50_test
from preprocessing import preprocessingCIFAR, preprocessingMINST, preprocessingNews, preprocessingAdult, \
    preprocessingCIFAR_GAN
from train import trainTarget


def load_data(data_name):
    with np.load(data_name) as f:
        train_x, train_y = [f['arr_%d' % i] for i in range(len(f.files))]
    return train_x, train_y


def shuffleAndSplitData_(dataX, dataY, cluster):
    c = list(zip(dataX, dataY))
    random.shuffle(c)
    dataX, dataY = zip(*c)

    data_x = np.array(dataX[:cluster])
    data_y = np.array(dataY[:cluster])
    datax = np.array(dataX[cluster:cluster*2])
    datay = np.array(dataY[cluster:cluster*2])

    return data_x, data_y ,datax ,datay


def initializeData(dataset,
                   orginialDatasetPath,
                   dataFolderPath='./data/'):
    if dataset == 'CIFAR10':

        print("Loading data")
        # dataX, dataY, testX, testY = readCIFAR10()

        # # dataX, dataY = readcifar10_StyleGAN_all()
        # # print("Preprocessing data")
        # #
        # cluster = 10520
        dataPath = dataFolderPath + dataset + '/Preprocessed'
        # #
        # toTrainData, toTrainLabel, shadowData, shadowLabel, toTestData, toTestLabel, shadowTestData, shadowTestLabel = \
        #     shuffleAndSplitData(dataX, dataY, cluster)

        # toTrainDataSave, toTestDataSave = \
        #     preprocessingCIFAR(toTrainData, toTestData)

        # shadowDataSave, shadowTestDataSave = \
        #     preprocessingCIFAR(shadowData, shadowTestData)

        # StyleGAN_adv1
        shadow_totraindata, shadow_totrainlable = readcifar10_StyleGAN_shadow()
        target_totraindata, target_totrainlable = readcifar10_StyleGAN_target()
        
        shadow_totestdata, shadowTestLabel = readcifar10_shadow_test()
        target_totestdata, toTestLabel = readcifar10_target_test()
        or_shadow_x, or_shadow_y = readcifar10_shadow_train()
        or_target_x, or_target_y = readcifar10_target_train()
        
        shadow_totraindata = shadow_totraindata.astype(float)
        target_totraindata = target_totraindata.astype(float)
        
        cluster = 5260
        shadow_totraindata, shadow_totrainlable, _, _ = shuffleAndSplitData_(shadow_totraindata, shadow_totrainlable, cluster)
        target_totraindata, target_totrainlable, _, _ = shuffleAndSplitData_(target_totraindata, target_totrainlable, cluster)
        
        shadowTrainData, shadowTrainLabel, _, _ = shuffleAndSplitData_(or_shadow_x, or_shadow_y, cluster)  # 加的
        targetTrainData, targetTrainLabel, _, _ = shuffleAndSplitData_(or_target_x, or_target_y, cluster)  # 加的
        shadow_totraindata = np.reshape(shadow_totraindata, (len(shadow_totraindata), -1)) # 
        target_totraindata = np.reshape(target_totraindata, (len(target_totraindata), -1)) #
        
        shadow_totraindata = np.concatenate((shadow_totraindata, shadowTrainData), axis=0) #
        shadowLabel = np.concatenate((shadow_totrainlable, shadowTrainLabel), axis=0) #
        target_totraindata = np.concatenate((target_totraindata, targetTrainData), axis=0) #
        toTrainLabel = np.concatenate((target_totrainlable, targetTrainLabel), axis=0) #
        
        
        # or_shadow_x, or_shadow_y = readcifar10_shadow_train()
        # or_target_x, or_target_y = readcifar10_target_train()
        
        toTrainDataSave = preprocessingCIFAR(target_totraindata)
        toTestDataSave = preprocessingCIFAR(target_totestdata)

        shadowDataSave = preprocessingCIFAR(shadow_totraindata)
        shadowTestDataSave = preprocessingCIFAR(shadow_totestdata)
        

    elif dataset == 'CIFAR100':
        #
        print("Loading data")
        dataX, dataY, testX, testY = readCIFAR100(orginialDatasetPath)
        print("Preprocessing data")

        cluster = 10500
        dataPath = dataFolderPath + dataset + '/Preprocessed'

        toTrainData, toTrainLabel, shadowData, shadowLabel, toTestData, toTestLabel, shadowTestData, shadowTestLabel = \
            shuffleAndSplitData(dataX, dataY, cluster)

        toTrainDataSave, toTestDataSave = \
            preprocessingCIFAR(toTrainData, toTestData)

        shadowDataSave, shadowTestDataSave = \
            preprocessingCIFAR(shadowData, shadowTestData)
        #————————————————-------adv1————————————————————————
        # print("Loading data")
        # dataPath = dataFolderPath + dataset + '/Preprocessed'
        # shadow_totraindata, shadow_totrainlable = readCIFAR100_StyleGAN_shadow()
        # target_totraindata, target_totrainlable = readCIFAR100_StyleGAN_target()
        #
        # datax,datay = readCIFAR100_StyleGAN_test()
        #
        # cluster=10520
        # target_totestdata, target_totestlable, shadow_totestdata, shadow_totestlable = shuffleAndSplitData_(datax,datay,cluster)
        # print(target_totestdata.shape)
        # or_target_x, or_target_y,or_shadow_x, or_shadow_y = readCIFAR100_StyleGAN_ori()

        # shadow_totraindata = shadow_totraindata.astype(float)
        # target_totraindata = target_totraindata.astype(float)

        or_shadow_x,shadow_totestdata= preprocessingCIFAR_GAN(or_shadow_x,shadow_totestdata)
        or_target_x,target_totestdata= preprocessingCIFAR_GAN(or_target_x,target_totestdata)

        target_totraindata,shadow_totraindata = preprocessingCIFAR_GAN(target_totraindata,shadow_totraindata)

    elif dataset == 'MINST':
        ######____________ori_adv1_______________________
        print("Loading data")
        data_path = './data/MINST'
        # dataX, dataY, testX, testY = readMNISTBin()
        dataX, dataY, testX, testY = readMINST(data_path)
        print("Preprocessing data")

        cluster = 10520
        dataPath = dataFolderPath + dataset + '/Preprocessed'

        toTrainData, toTrainLabel, shadowData, shadowLabel, toTestData, toTestLabel, shadowTestData, shadowTestLabel = \
            shuffleAndSplitData(dataX, dataY, cluster)
        #Bin_process
        # toTrainDataSave, toTestDataSave = \
        #     preprocessingCIFAR_GAN(toTrainData, toTestData)
        #
        # shadowDataSave, shadowTestDataSave = \
        #     preprocessingCIFAR_GAN(shadowData, shadowTestData)
        toTrainDataSave,toTestDataSave = preprocessingMINST(toTrainData,toTestData)
        shadowDataSave,shadowTestDataSave = preprocessingMINST(shadowData,shadowTestData)
        #______________________________adv1_gan________________________
        # print("Loading data")
        # dataPath = dataFolderPath + dataset + '/Preprocessed'
        # shadow_totraindata, shadow_totrainlable = readMINST_StyleGAN_shadow()
        # target_totraindata, target_totrainlable = readMINST_StyleGAN_target()
        #
        # datax,datay = readMINTS_StyleGAN_test()
        #
        # cluster=10520
        # target_totestdata, target_totestlable, shadow_totestdata, shadow_totestlable = shuffleAndSplitData_(datax,datay,cluster)
        # print(target_totestdata.shape)
        # or_target_x, or_target_y,or_shadow_x, or_shadow_y = readMINTS_StyleGAN_ori()
        #
        # shadow_totraindata = shadow_totraindata.astype(float)
        # target_totraindata = target_totraindata.astype(float)
        #
        # or_shadow_x,shadow_totestdata= preprocessingMINST(or_shadow_x,shadow_totestdata)
        # or_target_x,target_totestdata= preprocessingMINST(or_target_x,target_totestdata)
        #
        # target_totraindata,shadow_totraindata = preprocessingMINST(target_totraindata,shadow_totraindata)



    elif dataset == 'News':

        newsgroups_train, newsgroups_test = readNews(orginialDatasetPath)
        print("Preprocessing data")

        cluster = 4500
        dataPath = dataFolderPath + dataset + '/Preprocessed'

        toTrainData, toTrainLabel, shadowData, shadowLabel, toTestData, toTestLabel, shadowTestData, shadowTestLabel = \
            shuffleAndSplitData(newsgroups_train, newsgroups_test, cluster)

        toTrainDataSave, toTestDataSave = \
            preprocessingNews(toTrainData, toTestData)

        shadowDataSave, shadowTestDataSave = \
            preprocessingNews(shadowData, shadowTestData)

    elif dataset == 'LFW':

        print("Loading data")
        dataX, dataY, testX, testY = readFLW(orginialDatasetPath)
        print("Preprocessing data")


        print(dataX.shape)

        cluster = 420
        dataPath = dataFolderPath + dataset + '/Preprocessed'

        toTrainData, toTrainLabel, shadowData, shadowLabel, toTestData, toTestLabel, shadowTestData, shadowTestLabel = \
            shuffleAndSplitData(dataX, dataY, cluster)

        toTrainDataSave, toTestDataSave, shadowDataSave, shadowTestDataSave = \
            toTrainData, toTestData, shadowData, shadowTestData

    elif dataset == 'Adult':

        print("loading data")
        dataX, dataY, testX, testY = readAdult(orginialDatasetPath)

        dataX = np.concatenate((dataX, testX), axis=0)
        dataY = np.concatenate((dataY, testY), axis=0)

        print("processing data")

        cluster = 5600
        dataPath = dataFolderPath + dataset + '/Preprocessed'

        toTrainData, toTrainLabel, shadowData, shadowLabel, toTestData, toTestLabel, shadowTestData, shadowTestLabel = \
            shuffleAndSplitData(dataX, dataY, cluster)

        toTrainDataSave, toTestDataSave = \
            preprocessingAdult(toTrainData, toTestData)

        shadowDataSave, shadowTestDataSave = \
            preprocessingAdult(shadowData, shadowTestData)

    elif dataset == 'Purchase50':

        print("Loading data")
        # dataX, dataY = readPurchase50()
        toTrainData, toTrainLabel, shadowData, shadowLabel = readPurchase50_train()
        toTestData, toTestLabel, shadowTestData, shadowTestLabel = readPurchase50_test()
        print("Preprocessing data")

        cluster = 10005
        dataPath = dataFolderPath + dataset + '/Preprocessed'

        # toTrainData, toTrainLabel, shadowData, shadowLabel, toTestData, toTestLabel, shadowTestData, shadowTestLabel = \
        #     shuffleAndSplitData(dataX, dataY, cluster)

        toTrainDataSave, toTestDataSave, shadowDataSave, shadowTestDataSave = \
            toTrainData, toTestData, shadowData, shadowTestData
        # dataPath = dataFolderPath + dataset + '/Preprocessed'
        #
        # shadow_totraindata, shadow_totrainlable = readPurchase50_CTGAN_shadow()
        # target_totraindata, target_totrainlable = readPurchase50_CTGAN_target()
        #
        # target_totestdata,target_totestlable,shadow_totestdata, shadow_totestlable = readPurchase50_test()
        # or_target_x, or_target_y,or_shadow_x, or_shadow_y = readPurchase50_train()
    elif dataset == 'Purchase100':

        print("Loading data")
        print("Preprocessing data")
        datax,datay = readPurchase100()
        print(datax.shape)
        cluster=10005
        toTrainData, toTrainLabel, shadowData, shadowLabel,toTestData, toTestLabel,\
        shadowTestData, shadowTestLabel = shuffleAndSplitData(datax,datay,cluster)

        # toTrainData, toTrainLabel, shadowData, shadowLabel = readPurchase100_train()
        # toTestData, toTestLabel, shadowTestData, shadowTestLabel = readPurchase100_test()

        # cluster = 10005
        dataPath = dataFolderPath + dataset + '/Preprocessed'

        # toTrainData, toTrainLabel, shadowData, shadowLabel, toTestData, toTestLabel, shadowTestData, shadowTestLabel = \
        #     shuffleAndSplitData(dataX, dataY, cluster)

        toTrainDataSave, toTestDataSave, shadowDataSave, shadowTestDataSave = \
            toTrainData, toTestData, shadowData, shadowTestData


        # # defense process adv
        #
        #  dataPath = dataFolderPath + dataset + '/Preprocessed'
        #  shadow_totraindata, shadow_totrainlable = readPurchase100_CTGAN_shadow()
        #  target_totraindata, target_totrainlable = readPurchase100_CTGAN_target()
        #
        #  target_totestdata,target_totestlable,shadow_totestdata, shadow_totestlable = readPurchase100_test()
        #  or_target_x, or_target_y,or_shadow_x, or_shadow_y = readPurchase100_train()
    elif dataset == 'Purchase10':

        print("Loading data")
        print("Preprocessing data")

        toTrainData, toTrainLabel, shadowData, shadowLabel = readPurchase10_train()
        toTestData, toTestLabel, shadowTestData, shadowTestLabel = readPurchase10_test()
        # # dataX,dataY = readPurchase10()
        # #
        # # cluster = 10005
        dataPath = dataFolderPath + dataset + '/Preprocessed'
        # #
        # # toTrainData, toTrainLabel, shadowData, shadowLabel, toTestData, toTestLabel, shadowTestData, shadowTestLabel = \
        # #     shuffleAndSplitData(dataX, dataY, cluster)
        #
        toTrainDataSave, toTestDataSave, shadowDataSave, shadowTestDataSave = \
            toTrainData, toTestData, shadowData, shadowTestData

        # defense process adv

        # dataPath = dataFolderPath + dataset + '/Preprocessed'
        # shadow_totraindata, shadow_totrainlable = readPurchase10_CTGAN_shadow()
        # target_totraindata, target_totrainlable = readPurchase10_CTGAN_target()
        #
        # target_totestdata,target_totestlable,shadow_totestdata, shadow_totestlable = readPurchase10_test()
        # or_target_x, or_target_y,or_shadow_x, or_shadow_y = readPurchase10_train()

    elif dataset == 'Purchase20':

        print("Loading data")
        print("Preprocessing data")
        #
        toTrainData, toTrainLabel, shadowData, shadowLabel = readPurchase20_train()
        toTestData, toTestLabel, shadowTestData, shadowTestLabel = readPurchase20_test()
        # dataX,dataY = readPurchase20()

        # cluster = 10005
        dataPath = dataFolderPath + dataset + '/Preprocessed'
        # #
        # toTrainData, toTrainLabel, shadowData, shadowLabel, toTestData, toTestLabel, shadowTestData, shadowTestLabel = \
        #     shuffleAndSplitData(dataX, dataY, cluster)

        toTrainDataSave, toTestDataSave, shadowDataSave, shadowTestDataSave = \
            toTrainData, toTestData, shadowData, shadowTestData

        # defense process adv

        # dataPath = dataFolderPath + dataset + '/Preprocessed'
        # shadow_totraindata, shadow_totrainlable = readPurchase20_CTGAN_shadow()
        # target_totraindata, target_totrainlable = readPurchase20_CTGAN_target()
        #
        # target_totestdata,target_totestlable,shadow_totestdata, shadow_totestlable = readPurchase20_test()
        # or_target_x, or_target_y,or_shadow_x, or_shadow_y = readPurchase20_train()

    elif dataset == 'Location':
        #
        # print("Loading data")
        # datax,datay = readLocation()
        # print("Preprocessing data")
        #
        # cluster = 1250
        # dataPath = dataFolderPath + dataset + '/Preprocessed'
        #
        # toTrainData, toTrainLabel, shadowData, shadowLabel, toTestData, toTestLabel, shadowTestData, shadowTestLabel = \
        #     shuffleAndSplitData(datax, datay, cluster)
        #
        # toTrainDataSave, toTestDataSave, shadowDataSave, shadowTestDataSave = \
        #     toTrainData, toTestData, shadowData, shadowTestData

        # kmeans_process_data
        #
        toTrainData, toTrainLabel, shadowData, shadowLabel = readLocation_train()
        toTestData, toTestLabel, shadowTestData, shadowTestLabel = readLocation_test()

        # cluster = 10005
        dataPath = dataFolderPath + dataset + '/Preprocessed'

        # toTrainData, toTrainLabel, shadowData, shadowLabel, toTestData, toTestLabel, shadowTestData, shadowTestLabel = \
        #     shuffleAndSplitData(dataX, dataY, cluster)

        toTrainDataSave, toTestDataSave, shadowDataSave, shadowTestDataSave = \
            toTrainData, toTestData, shadowData, shadowTestData

        # ## ctgan process data
        #   dataPath = dataFolderPath + dataset + '/Preprocessed'
        #   shadow_totraindata, shadow_totrainlable = readLocation_CTGAN_shadow()
        #   target_totraindata, target_totrainlable = readLocation_CTGAN_target()
        #   dataX, dataY = readLocation_test()
        #   cluster = 1250
        #   target_totestdata, target_totestlable, shadow_totestdata, shadow_totestlable = shuffleAndSplitData_(dataX,dataY,cluster)
        #   or_target_x, or_target_y,or_shadow_x, or_shadow_y = readLocation_train()

    elif dataset == 'Purchase2':

        print("Loading data")
        print("Preprocessing data")

        toTrainData, toTrainLabel, shadowData, shadowLabel = readPurchase2_train()
        toTestData, toTestLabel, shadowTestData, shadowTestLabel =readPurchase2_test()

        # cluster = 10005
        dataPath = dataFolderPath + dataset + '/Preprocessed'

        # toTrainData, toTrainLabel, shadowData, shadowLabel, toTestData, toTestLabel, shadowTestData, shadowTestLabel = \
        #     shuffleAndSplitData(dataX, dataY, cluster)

        toTrainDataSave, toTestDataSave, shadowDataSave, shadowTestDataSave = \
            toTrainData, toTestData, shadowData, shadowTestData

        # dataPath = dataFolderPath + dataset + '/Preprocessed'
        # shadow_totraindata, shadow_totrainlable = readPurchase2_CTGAN_shadow()
        # target_totraindata, target_totrainlable = readPurchase2_CTGAN_target()
        #
        # target_totestdata, target_totestlable, shadow_totestdata, shadow_totestlable = readPurchase2_test()
        # or_target_x, or_target_y, or_shadow_x, or_shadow_y = readPurchase2_train()


    try:
        os.makedirs(dataPath)
    except OSError:
        pass

    np.savez(dataPath + '/targetTrain.npz', toTrainDataSave, toTrainLabel)
    np.savez(dataPath + '/targetTest.npz', toTestDataSave, toTestLabel)
    np.savez(dataPath + '/shadowTrain.npz', shadowDataSave, shadowLabel)
    np.savez(dataPath + '/shadowTest.npz', shadowTestDataSave, shadowTestLabel)

    # np.savez(dataPath + '/GAN_targetTrain.npz', target_totraindata, target_totrainlable)
    # np.savez(dataPath + '/targetTest.npz', target_totestdata, target_totestlable)
    # np.savez(dataPath + '/GAN_shadowTrain.npz', shadow_totraindata, shadow_totrainlable)
    # np.savez(dataPath + '/shadowTest.npz', shadow_totestdata, shadow_totestlable)
    # np.savez(dataPath + '/targetTrain.npz', or_target_x, or_target_y)
    # np.savez(dataPath + '/shadowTrain.npz', or_shadow_x, or_shadow_y)

    print("Preprocessing finished\n\n")


def initializeTargetModel(dataset,
                          num_epoch,
                          dataFolderPath='./data/',
                          modelFolderPath='./model/',
                          classifierType='cnn',
                          model='DP'):

    dataPath = dataFolderPath + dataset + '/Preprocessed'

    attackerModelDataPath = dataFolderPath + dataset+'/attackerModelData'

    modelPath = modelFolderPath + dataset

    try:
        # os.makedirs(attackerModelDataPath)
        os.makedirs(modelPath)
    except OSError:
        pass
    print("Training the Target model for {} epoch".format(num_epoch))

    targetTrain, targetTrainLabel = load_data(dataPath + '/targetTrain.npz')
    # or_targetTrain, or_targetTrainLabel = load_data(dataPath + '/targetTrain.npz')
    targetTest, targetTestLabel = load_data(dataPath + '/targetTest.npz')

    attackModelDataTarget, attackModelLabelsTarget, targetModelToStore = trainTarget(classifierType,
                                                                                     targetTrain,
                                                                                     targetTrainLabel,
                                                                                     # or_Train = or_targetTrain,
                                                                                     # or_TrainLabel =or_targetTrainLabel,
                                                                                     X_test=targetTest,
                                                                                     y_test=targetTestLabel,
                                                                                     splitData=False,
                                                                                     inepochs=num_epoch,
                                                                                     batch_size=100,
                                                                                     model=model)

    np.savez(attackerModelDataPath + '/targetModelData.npz',
             attackModelDataTarget, attackModelLabelsTarget)
    # np.savez('dataPurchase/Purchase50/10005/targetModelData.npz',
    #          attackModelDataTarget, attackModelLabelsTarget)

    torch.save(targetModelToStore, modelPath + '/targetModel.pth')

    return attackModelDataTarget, attackModelLabelsTarget


def initializeShadowModel(dataset, num_epoch, dataFolderPath='./data/', modelFolderPath='./model/',
                          classifierType='cnn', model='DP'):
    dataPath = dataFolderPath + dataset + '/Preprocessed'

    attackerModelDataPath = dataFolderPath + dataset + '/attackerModelData'

    modelPath = modelFolderPath + dataset


    try:
        os.makedirs(modelPath)
    except OSError:
        pass
    print("Training the Shadow model for {} epoch".format(num_epoch))

    shadowTrainData, shadowTrainLabel = load_data(dataPath + '/shadowTrain.npz')
    # or_shadowTrain, or_shadowTrainLabel = load_data(dataPath + '/shadowTrain.npz')
    shadowTestData, shadowTestLabel = load_data(dataPath + '/shadowTest.npz')

    attackModelDataShadow, attackModelLabelsShadow, shadowModelToStore = trainTarget(classifierType,
                                                                                     shadowTrainData,
                                                                                     shadowTrainLabel,
                                                                                     # or_Train= or_shadowTrain,
                                                                                     # or_TrainLabel= or_shadowTrainLabel,
                                                                                     X_test=shadowTestData,
                                                                                     y_test=shadowTestLabel,
                                                                                     splitData=False,
                                                                                     inepochs=num_epoch,
                                                                                     batch_size=100,
                                                                                     model=model)
    
    np.savez(attackerModelDataPath + '/shadowModelData.npz',
             attackModelDataShadow, attackModelLabelsShadow)
    
    # np.savez('dataPurchase/Purchase50/10005/shadowModelData.npz',
    #          attackModelDataShadow, attackModelLabelsShadow)
    
    torch.save(shadowModelToStore, modelPath + '/shadowModel.pth')

    return attackModelDataShadow, attackModelLabelsShadow


def generateAttackData(dataset, classifierType, dataFolderPath, pathToLoadData, num_epoch, preprocessData,
                       trainTargetModel, trainShadowModel, model, topX=3):
    print(dataset, dataFolderPath, preprocessData)

    attackerModelDataPath = dataFolderPath + dataset +'/attackerModelData'

    if (preprocessData):
        initializeData(dataset, pathToLoadData)

    if (trainTargetModel):
        targetX, targetY = initializeTargetModel(dataset,
                                                 num_epoch,
                                                 classifierType=classifierType,
                                                 model=model)
    else:
        targetX, targetY = load_data(attackerModelDataPath + '/targetModelData.npz')

    if (trainShadowModel):
        shadowX, shadowY = initializeShadowModel(dataset,
                                                 num_epoch,
                                                 classifierType=classifierType,
                                                 model=model)
    else:
        shadowX, shadowY = load_data(attackerModelDataPath + '/shadowModelData.npz')

    targetX = clipDataTopX(targetX, top=topX)
    shadowX = clipDataTopX(shadowX, top=topX)

    return targetX, targetY, shadowX, shadowY

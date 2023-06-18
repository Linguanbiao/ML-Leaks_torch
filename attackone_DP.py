#！-*- coding:utf-8 -*-
"""
Created on 5 Dec 2018

@author: Wentao Liu, Ahmed Salem
"""
import argparse
from sklearn.metrics import roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets import fetch_20newsgroups
import os
import random
from sklearn.model_selection import train_test_split
import pickle
import numpy as np
import sys
from dataset import readCIFAR10
from initialization import generateAttackData
from preprocessing import preprocessingCIFAR, preprocessingNews,preprocessingAdult,preprocessingMINST
import torch
from torchvision.datasets import CIFAR10, CIFAR100, MNIST
import torchvision.transforms as transforms

from train import trainAttackModel

seed = 21312
sys.dont_write_bytecode = True
np.random.seed(seed)

parser = argparse.ArgumentParser()
parser.add_argument('--adv',  default='1', help='Which adversary 1, 2, or 3')
parser.add_argument('--dataset', default='CIFAR10',
                    help='Which dataset to use (CIFAR10,CIFAR100,MINST,LFW,Adult,or News)')
parser.add_argument('--classifierType', default='cnn',
                    help='Which classifier cnn or cnn')
parser.add_argument('--dataset2', default='News',
                    help='Which second dataset for adversary 2 (CIFAR10 or News)')
parser.add_argument('--classifierType2', default='nn',
                    help='Which classifier cnn or nn')
parser.add_argument('--dataFolderPath', default='./data',
                    help='Path to store data')
parser.add_argument('--pathToLoadData', default='./data/png2npy_2',# cifar-10-batches-py,cifar-100-python,MINST,lfw,Adult,News
                    help='Path to load dataset from')
parser.add_argument('--num_epoch', type=int, default=50,
                    help='Number of epochs to train shadow/target models')
parser.add_argument('--preprocessData',  default=True,action='store_true',
                    help='Preprocess the data, if false then load preprocessed data')
parser.add_argument('--trainTargetModel', default=True,action='store_true',
                    help='Train a target model, if false then load an already trained model')
parser.add_argument('--trainShadowModel', default=True,action='store_true',
                    help='Train a shadow model, if false then load an already trained model')
parser.add_argument('--model', default='normal',help='train a target model/shadow model use DP or not')
opt = parser.parse_args()


def save_attack_data(Dataser1X, Dataset1Y, Dataset2X, Dataset2Y, dataset, cluster, dataFolderPath) :
    attackerModelDataPath = dataFolderPath + dataset + '/' + str(cluster)
    try:
        os.makedirs(attackerModelDataPath)
    except OSError:
        pass
    np.savez(attackerModelDataPath + '/targetModelData.npz',
             Dataser1X, Dataset1Y)
    np.savez(attackerModelDataPath + '/shadowModelData.npz',
             Dataset2X, Dataset2Y)


def attackerOne(dataset='CIFAR10',
                classifierType='cnn',
                dataFolderPath='./data/',
                pathToLoadData='./data',
                num_epoch=200,
                preprocessData=False,
                trainTargetModel=False,
                trainShadowModel=False,
                model='DP'):

    targetX, targetY, shadowX, shadowY = \
        generateAttackData(dataset,
                           classifierType,
                           dataFolderPath,
                           pathToLoadData,
                           num_epoch,
                           preprocessData,
                           trainTargetModel,
                           trainShadowModel,
                           model)

    print("Training the attack model for the first adversary")
    attackModel_one = trainAttackModel(targetX, targetY, shadowX, shadowY,model='normal')
    modelPath = './model/' + dataset
    torch.save(attackModel_one, modelPath + '/attackModel_one.pth')



def attackerTwo(dataset1='CIFAR10',
                dataset2='News',
                classifierType1='cnn',
                classifierType2='nn',
                dataFolderPath='./data/',
                pathToLoadData='./data/cifar-10-batches-py-official',
                num_epoch=50,
                preprocessData=True,
                trainTargetModel=True,
                trainShadowModel=True):

    Dataset1X, Dataset1Y, _, _ = \
        generateAttackData(dataset1,
                           classifierType1,
                           dataFolderPath,
                           pathToLoadData,
                           num_epoch,
                           preprocessData,
                           trainTargetModel,
                           trainShadowModel)

    Dataset2X, Dataset2Y, _, _ = \
        generateAttackData(dataset2,
                           classifierType2,
                           dataFolderPath,
                           pathToLoadData,
                           num_epoch,
                           preprocessData,
                           trainTargetModel,
                           trainShadowModel)

    print("Training the attack model for the second adversary")
    trainAttackModel(Dataset1X, Dataset1Y, Dataset2X, Dataset2Y)



def attackerThree(dataset='CIFAR10',
                  classifierType='cnn',
                  dataFolderPath='./data/',
                  pathToLoadData='./data/cifar-10-batches-py-official',
                  num_epoch=50,
                  preprocessData=True,
                  trainTargetModel=True):

    targetX, targetY, _, _ = \
        generateAttackData(dataset,
                           classifierType,
                           dataFolderPath,
                           pathToLoadData,
                           num_epoch,
                           preprocessData,
                           trainTargetModel,
                           trainShadowModel=False,
                           topX=1)

    print('AUC = {}'.format(roc_auc_score(targetY, targetX)))


if opt.adv == '1':

    attackerOne(dataset=opt.dataset,
                classifierType=opt.classifierType,
                dataFolderPath=opt.dataFolderPath,
                pathToLoadData=opt.pathToLoadData,
                num_epoch=opt.num_epoch,
                preprocessData=opt.preprocessData,
                trainTargetModel=opt.trainTargetModel,
                trainShadowModel=opt.trainShadowModel,
                model=opt.model)

elif opt.adv == '2':

    attackerTwo(dataset1=opt.dataset,
                dataset2=opt.dataset2,
                classifierType1=opt.classifierType,
                classifierType2=opt.classifierType2,
                dataFolderPath=opt.dataFolderPath,
                pathToLoadData=opt.pathToLoadData,
                num_epoch=opt.num_epoch,
                preprocessData=opt.preprocessData,
                trainTargetModel=opt.trainTargetModel,
                trainShadowModel=opt.trainShadowModel)

elif opt.adv == '3':

    attackerThree(dataset=opt.dataset,
                  classifierType=opt.classifierType,
                  dataFolderPath=opt.dataFolderPath,
                  pathToLoadData=opt.pathToLoadData,
                  num_epoch=opt.num_epoch,
                  preprocessData=opt.preprocessData,
                  trainTargetModel=opt.trainTargetModel)

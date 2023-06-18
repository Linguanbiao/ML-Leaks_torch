import argparse
import os

import torch.nn as nn
import torch.nn.functional as func
import torch
import torch.optim as optim
from torchvision import datasets, transforms
import pickle
import numpy as np
import random
from sklearn.metrics import classification_report
from sklearn.metrics import multilabel_confusion_matrix

import matplotlib.pyplot as plt

def seed_torch(seed=1029):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True

seed_torch()

class CNN_Model(nn.Module):
    def __init__(self, n_in, n_hidden, n_out):
        super(CNN_Model, self).__init__()
        self.conv1 = nn.Sequential(
                        nn.Conv2d(n_in[1], 32, kernel_size = (5, 5), padding = 2, stride = 1),
                        nn.ReLU()
                    )
        self.maxp1 = nn.MaxPool2d(kernel_size = (2, 2))
        self.conv2 = nn.Sequential(
                        nn.Conv2d(32, 32, kernel_size = (5, 5), padding = 0, stride = 1),
                        nn.ReLU()
                    )

        self.maxp2 = nn.MaxPool2d(kernel_size = (2, 2))
        self.fc1 = nn.Sequential(
                        nn.Linear(32 * 6 * 6, n_hidden),
                        nn.Tanh()
                    )

        self.fc2 = nn.Sequential(
                        nn.Linear(n_hidden, n_out)
                    )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.)
            elif isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight, gain = nn.init.calculate_gain('relu'))
                nn.init.constant_(m.bias, 0.)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxp1(x)
        x = self.conv2(x)
        x = self.maxp2(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

class NN_Model(nn.Module):
    def __init__(self, n_in, n_hidden, n_out):
        super(NN_Model,self).__init__()
        self.fc1 = nn.Sequential(
                            nn.Linear(n_in[1], n_hidden),
                            nn.Tanh()
                        )
        self.fc2 = nn.Sequential(
                            nn.Linear(n_hidden, n_out)
                        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)

        return x

class Softmax_Model(nn.Module):

    def __init__(self, n_in, n_out):
        super(Softmax_Model, self).__init__()
        self.fc1 = nn.Sequential(
                            nn.Linear(n_in[1], n_out)
                )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.)

    def forward(self, x):
        x = self.fc1(x)

        return x

#定义交叉熵损失
def cross_entropy_loss(output, target):
	return -torch.sum(output.log() * target) / output.shape[0]
#定义kl损失
def KL_loss(output, target):
	return (torch.sum(target.log() * target) - torch.sum(output.log() * target)) / output.shape[0]



#训练模型的过程
def train(args, model, device, X, y, optimizer, criterion, epoch):
    model.train()
    batch_idx = 0
    for data, target in iterate_minibatches(X, y, args.batch_size):
        data, target = torch.tensor(data).type(torch.float), torch.tensor(target).type(torch.long)  #.type(torch.float)
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        # if batch_idx % args.log_interval == 0:
        #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        #         epoch, batch_idx * len(data), len(X),
        #                100. * batch_idx * len(data) / len(X), loss.item()))
        # batch_idx += 1

    return model

#训练学生模型的过程
def train_S(args, model, device, X, y, optimizer, criterion, epoch, Teacher_model):
    model.train()
    batch_idx = 0
    for data, target in iterate_minibatches(X, y, args.batch_size):
        data, target = torch.tensor(data).type(torch.float), torch.tensor(target).type(torch.long)  # .type(torch.float)
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        batch_y_teacher = torch.softmax(Teacher_model(data) / args.KD_temperature, dim=1).detach()
        loss = (1 - args.KD_loss_lambda) * criterion[0](output, target) + \
                args.KD_loss_lambda * args.KD_temperature ** 2 * criterion[1](torch.softmax(output / args.KD_temperature, dim=1),batch_y_teacher)
        loss.backward()
        optimizer.step()
        # if batch_idx % args.log_interval == 0:
        #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        #         epoch, batch_idx * len(data), len(X),
        #                100. * batch_idx * len(data) / len(X), loss.item()))
        # batch_idx += 1

    return model

#进行测试
def test(args, model, device,  X_test, y_test, criterion, Flag1 = "Test", Flag2 = True):
    if(Flag2):
        model.eval()
    test_loss = 0
    correct = 0
    pred_y_lable = []
    pre_y_distribution = []
    with torch.no_grad():
        for data, target in iterate_minibatches( X_test, y_test, batch_size=args.test_batch_size, shuffle=False):
            data, target = torch.tensor(data).type(torch.float), torch.tensor(target).type(torch.long)  #.type(torch.float)
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

            pred_y_lable.append(torch.max(output, 1)[1].data.cpu().numpy())
            pre_y_distribution.append(output.detach().cpu().numpy())   # no_softmax
    pred_y_lable = np.concatenate(pred_y_lable)        #预测标签
    pre_y_distribution = np.concatenate(pre_y_distribution)
    test_loss /= len(X_test)/args.test_batch_size

    # print('\n {} set: Average loss: {:.7f}, Accuracy: {}/{} ({:.2f}%)\n'.format(Flag1, test_loss, correct, len(X_test), 100. * correct / len(X_test)))

    return test_loss, correct/len(X_test), pred_y_lable, pre_y_distribution


def iterate_minibatches(inputs, targets, batch_size, shuffle=True):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)

    start_idx = None
    for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batch_size]
        else:
            excerpt = slice(start_idx, start_idx + batch_size)
        yield inputs[excerpt], targets[excerpt]

    if start_idx is not None and start_idx + batch_size < len(inputs):
        excerpt = indices[start_idx + batch_size:] if shuffle else slice(start_idx + batch_size, len(inputs))
        yield inputs[excerpt], targets[excerpt]

def Train_model(args, device,  X, y, X_test, y_test):
    # models = Net().to(device)
    # models = vgg(args.size_model, in_channels, num_classes).to(device)
    # models = GoogLeNet().to(device)
    n_in = X.shape
    n_out = len(np.unique(y))
    n_hidden = 128
    model = CNN_Model(n_in, n_hidden, n_out).to(device)
    # model = NN_Model(n_in, n_hidden, n_out).to(device)
    # print(model)
    # optimizer = optim.SGD(models.parameters(), lr=args.lr, momentum=args.momentum)
    optimizer = optim.Adam(model.parameters(), lr=0.001) #, betas=(0.9, 0.999),eps=1e-08, weight_decay=1e-06
    criterion = nn.CrossEntropyLoss()
    tr_L, tr_A, te_L, te_A = [], [], [], []           #训练数据的损失、精确度，测试数据的损失和精确度
    for epoch in range(1, args.epochs + 1):
        model = train(args, model, device, X, y, optimizer, criterion, epoch)   #训练得到的一个模型
        tr_loss, tr_acc, tr_pred_y_lable, tr_pre_y_distribution = test(args, model, device, X, y, criterion,"Train", True)
        te_loss, te_acc, te_pred_y_lable, te_pre_y_distribution = test(args, model, device, X_test, y_test, criterion, "Test", True)
        tr_L.append(tr_loss)
        tr_A.append(tr_acc)
        te_L.append(te_loss)
        te_A.append(te_acc)

        print('Epoch:{} | Train set: Average loss: {:.7f}, Accuracy: {:.2f}  |  Test set: Average loss: {:.7f}, Accuracy: {:.4f}'.format(epoch, tr_loss, tr_acc, te_loss, te_acc))


    Paint(tr_A, te_A, tr_L, te_L, './'+args.model+'_acc-loss.png')

    if (args.save_models):
        torch.save(model, args.model+".pkl")

    # models = torch.load('./' + str(args.size_model)+'_'+args.data_type+".pkl")

    # print('More detailed results of attack_train:')
    # print(classification_report(y, tr_pred_y_lable))
    # mcm = multilabel_confusion_matrix(y, tr_pred_y_lable, labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    # print(mcm)
    #
    # print('More detailed results of attack_train:')
    # print(classification_report(y_test, te_pred_y_lable))
    # mcm = multilabel_confusion_matrix(y_test, te_pred_y_lable, labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    # print(mcm)

    return model, tr_pre_y_distribution, te_pre_y_distribution

def Train_model_student(args, device,  X, y ,TX, Ty, TX_test, Ty_test, Teacher_model):
    # models = Net().to(device)
    # models = vgg(args.size_model, in_channels, num_classes).to(device)
    # models = GoogLeNet().to(device)
    n_in = X.shape
    n_out = len(np.unique(y))
    n_hidden = 128
    # model = CNN_Model(n_in, n_hidden, n_out).to(device)
    model = NN_Model(n_in, n_hidden, n_out).to(device)
    # print(model)
    # optimizer = optim.SGD(models.parameters(), lr=args.lr, momentum=args.momentum)
    optimizer = optim.Adam(model.parameters(), lr=0.001) #lr=0.0002, betas=(0.9, 0.999),eps=1e-08, weight_decay=1e-06

    tr_L, tr_A, te_L, te_A = [], [], [], []
    for epoch in range(1, args.epochs + 1):
        criterion = [nn.CrossEntropyLoss(), cross_entropy_loss]
        # criterion = [nn.CrossEntropyLoss(), KL_loss]
        model = train_S(args, model, device, X, y, optimizer, criterion, epoch, Teacher_model)
        criterion = nn.CrossEntropyLoss()
        tr_loss, tr_acc, tr_pred_y_lable, tr_pre_y_distribution = test(args, model, device, TX, Ty, criterion, "Train", True)
        te_loss, te_acc, te_pred_y_lable, te_pre_y_distribution = test(args, model, device, TX_test, Ty_test, criterion, "Test", True)
        tr_L.append(tr_loss)
        tr_A.append(tr_acc)
        te_L.append(te_loss)
        te_A.append(te_acc)

        print('Epoch:{} | Teacher_Model： | Train set: Average loss: {:.7f}, Accuracy: {:.2f}  |  Test set: Average loss: {:.7f}, Accuracy: {:.2f}'.format(epoch, tr_loss, tr_acc, te_loss, te_acc))

    Paint(tr_A, te_A, tr_L, te_L, './'+args.model+'_acc-loss.png')

    if (args.save_models):
        torch.save(model, args.model+".pkl")

    # print('More detailed results of attack_train:')
    # print(classification_report(Ty, tr_pred_y_lable))
    # mcm = multilabel_confusion_matrix(Ty, tr_pred_y_lable, labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    # print(mcm)
    #
    # print('More detailed results of attack_train:')
    # print(classification_report(Ty_test, te_pred_y_lable))
    # mcm = multilabel_confusion_matrix(Ty_test, te_pred_y_lable, labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    # print(mcm)

    return model, tr_pre_y_distribution, te_pre_y_distribution

def load_data(data_name):
    with np.load(data_name) as f:
        train_x, train_y, test_x, test_y = [f['arr_%d' % i] for i in range(len(f.files))]
    return train_x, train_y, test_x, test_y

def clipDataTopX(dataToClip, top=3):
    res = [sorted(s, reverse=True)[0:top] for s in dataToClip]
    return np.array(res)

def paint_hist(args, targetX, Flag, x1=0, x2=1, y = 5000, b = 100):
    targetX = clipDataTopX(targetX, top=1)
    l = int(len(targetX)/2)
    targetX_1 = targetX[:l]
    targetX_0 = targetX[l:]
    img_1 = np.array(targetX_1)  # array是自己的一维数组，用np.array()将此数组变为numpy下的数组
    img_0 = np.array(targetX_0)
    plt.figure("lena")  # 定义了画板
    arr_1 = img_1.flatten()  # 若上面的array不是一维数组，flatten()将其变为一维数组，是numpy中的函数
    arr_0 = img_0.flatten()

    # hist函数可以直接绘制直方图
    # 参数有四个，第一个必选
    # arr: 需要计算直方图的一维数组
    # bins: 直方图的柱数，可选项，默认为10
    # density: 是否将得到的直方图向量归一化。默认为0
    # orientation: 决定了是采用纵轴代表频率还是横轴代表频率的展现形式
    # facecolor: 直方图颜色
    # alpha: 透明度
    # 返回值为n: 直方图向量，是否归一化由参数设定；bins: 返回各个bin的区间范围；patches: 返回每个bin里面包含的数据，是一个list
    n_1, bins_1, patches_1 = plt.hist(arr_1, b ,range =(0,1), density=False, facecolor='green', orientation='vertical', alpha=0.5, label = 'menbership')
    n_0, bins_0, patches_0 = plt.hist(arr_0, b ,range =(0,1), density=False, facecolor='red', orientation='vertical', alpha=0.5, label = 'non-menbership')
    # 添加x轴和y轴标签
    plt.xlabel('confidence value')
    plt.ylabel('number')
    plt.xlim(x1, x2)
    plt.ylim(0, y)
    # 添加标题
    plt.title(args.model)
    # 显示图例
    plt.legend()
    # 存储图片
    savepath = './' + args.model + Flag+ "_confidence value.png"
    plt.savefig(savepath, dpi=500, bbox_inches='tight')
    # 显示图形
    plt.show()

def Paint(train_acc, test_acc, train_loss, test_loss, path):
    # 设置输出的图片大小
    figsize = 6, 5
    figure, ax1 = plt.subplots(figsize=figsize)
    ax2 = ax1.twinx() #产生一个ax1的镜面坐标

    # 设置刻度标记的大小
    plt.tick_params(axis='both', which='minor', labelsize=14)

    ax1.set_xlabel('Epoch', fontsize=14)  # 设置x轴标题
    ax1.set_ylabel('Accuracy', color='g', fontsize=14)  # 设置Y1轴标题
    ax2.set_ylabel('Loss', color='b', fontsize=14)  # 设置Y2轴标题
    # plt.xlabel('iteration', fontdict={'family': 'Times New Roman', 'size': 14}) # 设置x轴标题
    # plt.ylabel('accuracy', fontdict={'family': 'Times New Roman', 'size': 14}) # 设置Y轴标题

    # 设置x轴范围
    x = np.arange(len(train_acc))

    # lns_1 = ax1.plot(x, train_acc, c='blue', label="train_acc")
    # lns_2 = ax2.plot(x, train_loss, c='red', label="train_loss")
    # lns_3 = ax1.plot(x, test_acc, c='blue', label="test_acc", linestyle='--')
    # lns_4 = ax2.plot(x, test_loss, c='red', label="test_loss", linestyle='--')

    lns_1 = ax1.plot(x, train_acc, c='blue')
    lns_2 = ax2.plot(x, train_loss, c='red')
    lns_3 = ax1.plot(x, test_acc, c='blue', linestyle='--')
    lns_4 = ax2.plot(x, test_loss, c='red', linestyle='--')

    # 合并图例
    lns = lns_1 + lns_2 + lns_3 + lns_4
    labels = ["train_acc", "train_loss", "test_acc", "test_loss"]

    # 添加每条曲线的标注
    plt.legend(lns, labels, loc=0)

    # 保存图片
    plt.savefig(path)

    #show函数展示出这个图，如果没有这行代码，则程序完成绘图，但看不到
    plt.show()

def Paint1(test_acc, test_loss, path):
    # 设置输出的图片大小
    figsize = 6, 5
    figure, ax1 = plt.subplots(figsize=figsize)
    ax2 = ax1.twinx() #产生一个ax1的镜面坐标

    # 设置刻度标记的大小
    plt.tick_params(axis='both', which='minor', labelsize=14)

    ax1.set_xlabel('Epoch', fontsize=14)  # 设置x轴标题
    ax1.set_ylabel('Accuracy', color='g', fontsize=14)  # 设置Y1轴标题
    ax2.set_ylabel('Loss', color='b', fontsize=14)  # 设置Y2轴标题
    # plt.xlabel('iteration', fontdict={'family': 'Times New Roman', 'size': 14}) # 设置x轴标题
    # plt.ylabel('accuracy', fontdict={'family': 'Times New Roman', 'size': 14}) # 设置Y轴标题

    # 设置x轴范围
    x = np.arange(len(test_acc))

    # lns_1 = ax1.plot(x, train_acc, c='blue', label="train_acc")
    # lns_2 = ax2.plot(x, train_loss, c='red', label="train_loss")
    # lns_3 = ax1.plot(x, test_acc, c='blue', label="test_acc", linestyle='--')
    # lns_4 = ax2.plot(x, test_loss, c='red', label="test_loss", linestyle='--')

    lns_3 = ax1.plot(x, test_acc, c='blue', linestyle='--')
    lns_4 = ax2.plot(x, test_loss, c='red', linestyle='--')

    # 合并图例
    lns = lns_3 + lns_4
    labels = ["test_acc", "test_loss"]

    # 添加每条曲线的标注
    plt.legend(lns, labels, loc=0)

    # 保存图片
    plt.savefig(path)

    #show函数展示出这个图，如果没有这行代码，则程序完成绘图，但看不到
    plt.show()

class CrossEntropy_L2(nn.Module):

    def __init__(self, model, m, l2_ratio):

        super(CrossEntropy_L2, self).__init__()
        self.model = model
        self.m = m
        self.w = 0.0
        self.l2_ratio = l2_ratio

    def forward(self, y_pred, y_test):

        criterion = nn.CrossEntropyLoss()
        loss = criterion(y_pred, y_test)

        for name in self.model.state_dict():

            if name.find('weight') != -1:
                self.w += torch.sum(torch.square(self.model.state_dict()[name]))

        loss = torch.add(torch.mean(loss), self.l2_ratio * self.w / self.m / 2)

        return loss

def attack_model(args, target_tr_pre_y_distribution, target_te_pre_y_distribution, shadow_tr_pre_y_distribution, shadow_te_pre_y_distribution):
    attack_test_x = []
    attack_test_y = []
    attack_train_x = []
    attack_train_y = []

    # target_tr_pre_y_lable = []
    # target_te_pre_y__lable = []
    # shadow_tr_pre_y__lable = []
    # shadow_te_pre_y__lable = []

    # target_tr_pre_y_distribution = nn.functional.softmax(target_tr_pre_y_distribution, dim=1)
    # target_tr_pre_y_lable.append(np.ones(len(target_tr_pre_y_distribution)))
    # target_te_pre_y_distribution = nn.functional.softmax(target_te_pre_y_distribution, dim=1)
    # target_te_pre_y__lable.append(np.zeros(len(target_te_pre_y_distribution)))
    # shadow_tr_pre_y_distribution = nn.functional.softmax(shadow_tr_pre_y_distribution, dim=1)
    # shadow_tr_pre_y__lable.append(np.ones(len(shadow_tr_pre_y_distribution)))
    # shadow_te_pre_y_distribution = nn.functional.softmax(shadow_te_pre_y_distribution, dim=1)
    # shadow_te_pre_y__lable.append(np.zeros(len(shadow_te_pre_y_distribution)))

    attack_test_x.append(nn.functional.softmax(torch.from_numpy(target_tr_pre_y_distribution), dim=1))
    attack_test_y.append(np.ones(len(target_tr_pre_y_distribution)))
    attack_test_x.append(nn.functional.softmax(torch.from_numpy(target_te_pre_y_distribution), dim=1))
    attack_test_y.append(np.zeros(len(target_te_pre_y_distribution)))
    attack_train_x.append(nn.functional.softmax(torch.from_numpy(shadow_tr_pre_y_distribution), dim=1))
    attack_train_y.append(np.ones(len(shadow_tr_pre_y_distribution)))
    attack_train_x.append(nn.functional.softmax(torch.from_numpy(shadow_te_pre_y_distribution), dim=1))
    attack_train_y.append(np.zeros(len(shadow_te_pre_y_distribution)))

    attack_test_x = np.vstack(attack_test_x)
    attack_test_y = np.concatenate(attack_test_y)
    attack_test_x = attack_test_x.astype('float32')
    attack_test_y = attack_test_y.astype('int32')
    attack_train_x = np.vstack(attack_train_x)
    attack_train_y = np.concatenate(attack_train_y)
    attack_train_x = attack_train_x.astype('float32')
    attack_train_y = attack_train_y.astype('int32')

    paint_hist(args, attack_train_x, '_train_x')
    paint_hist(args, attack_test_x, '_test_x')
    attack_train_x = clipDataTopX(attack_train_x, top=3)
    attack_test_x = clipDataTopX(attack_test_x, top=3)

    n_in = attack_train_x.shape
    n_out = len(np.unique(attack_train_y))

    model = Softmax_Model(n_in, n_out).to(device)
    # print(model)
    # optimizer = optim.SGD(models.parameters(), lr=args.lr, momentum=args.momentum)
    # optimizer = optim.Adam(models.parameters(), lr=0.02, betas=(0.9, 0.999),eps=1e-08, weight_decay=1e-06)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    # m = n_in[0]
    # criterion = CrossEntropy_L2(models, m, 1e-7).to(device)

    tr_L, tr_A, te_L, te_A = [], [], [], []

    for epoch in range(1, args.epochs + 51):
        model = train(args, model, device, attack_train_x, attack_train_y, optimizer, criterion, epoch)
        tr_loss, tr_acc, tr_pred_y_lable, tr_pre_y_distribution = test(args, model, device, attack_train_x, attack_train_y, criterion, "Train", True)
        te_loss, te_acc, te_pred_y_lable, te_pre_y_distribution = test(args, model, device, attack_test_x, attack_test_y, criterion, "Test", True)
        tr_L.append(tr_loss)
        tr_A.append(tr_acc)
        te_L.append(te_loss)
        te_A.append(te_acc)

        print('Epoch:{} | Train set: Average loss: {:.7f}, Accuracy: {:.2f}  |  Test set: Average loss: {:.7f}, Accuracy: {:.2f}'.format(
                epoch, tr_loss, tr_acc, te_loss, te_acc))

    Paint(tr_A, te_A, tr_L, te_L, './'+args.model+'_acc-loss.png')
    print('More detailed results of attack_train:')
    print(classification_report(attack_train_y, tr_pred_y_lable))
    mcm = multilabel_confusion_matrix(attack_train_y, tr_pred_y_lable, labels=[0, 1])
    print(mcm)

    print('More detailed results of attack_test:')
    print(classification_report(attack_test_y, te_pred_y_lable))
    mcm = multilabel_confusion_matrix(attack_test_y, te_pred_y_lable, labels=[0, 1])
    print(mcm)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Example')
    parser.add_argument('--models', type=str, default='TTM',
                        help='the wide size of models')
    parser.add_argument('--data-type', type=str, default="purchase", #cifar10, purchase
                        help='the wide size of models')
    parser.add_argument('--KD', default= True,
                        help='train student model')
    parser.add_argument('--KD-loss-lambda', type=float, default=1,
                        help='KD-loss-lambda')
    parser.add_argument('--KD-temperature', type=int, default=1,
                        help='KD-loss-lambda')

    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=64, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-models', action='store_true', default=True,
                        help='For Saving the current Model')

    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    if(args.data_type == 'cifar10'):
        TTX, TTy, TTX_test, TTy_test = load_data("./teacher_targetTrain_Test_data.npz")
        STX, STy, SSX, SSy = load_data("./student_targetTrain_Test_data.npz")
        STX, STy, STX_test, STy_test = load_data("./shadowTrain_Test_data.npz")
    elif(args.data_type == 'purchase'):
        TTX, TTy, TTX_test, TTy_test = load_data("./purchase/purchase100/teacher_targetTrain_Test_data.npz")
        STX, STy, SSX, SSy = load_data("./purchase/purchase100/student_targetTrain_Test_data.npz")
        STX, STy, STX_test, STy_test = load_data("./purchase/purchase100/shadowTrain_Test_data.npz")
    elif (args.data_type == 'cifar100'):
        TTX, TTy, TTX_test, TTy_test = load_data("./cifar100/teacher_targetTrain_Test_data.npz")
        STX, STy, SSX, SSy = load_data("./cifar100/student_targetTrain_Test_data.npz")
        STX, STy, STX_test, STy_test = load_data("./cifar100/shadowTrain_Test_data.npz")

    print("Training teacher_target model with 10000 train data and 10000 test data:")
    args.model = 'TTM'
    args.KD = False
    Ttarget_model, Ttarget_tr_pre_y_distribution, Ttarget_te_pre_y_distribution = Train_model(args, device, TTX, TTy, TTX_test, TTy_test)
    print("==========================================================================================================================================")

    print("Training student_target model with 10000 train data and 10000 test data:")
    args.model = 'STM'
    args.KD = True
    Starget_model, Starget_tr_pre_y_distribution, Starget_te_pre_y_distribution = Train_model_student(args, device, STX, STy, TTX, TTy, TTX_test, TTy_test, Ttarget_model)
    print("==========================================================================================================================================")

    print("Training teacher_shadow model with 10000 train data and 10000 test data:")
    args.model = 'TSM'
    args.KD = False
    Tshadow_model, Tshadow_tr_pre_y_distribution, Tshadow_te_pre_y_distribution = Train_model(args, device, STX, STy, STX_test, STy_test)
    print("==========================================================================================================================================")

    print("Training student_shadow model with 10000 train data and 10000 test data:")
    args.model = 'SSM'
    args.KD = True
    Sshadow_model, Sshadow_tr_pre_y_distribution, Sshadow_te_pre_y_distribution = Train_model_student(args, device, SSX, SSy, STX, STy, STX_test, STy_test, Tshadow_model)
    print("==========================================================================================================================================")

    print("Training TS-TT attack model with 20000 train data and 20000 test data:")
    args.model = 'TS-TT_AM'
    attack_model(args, Ttarget_tr_pre_y_distribution, Ttarget_te_pre_y_distribution, Tshadow_tr_pre_y_distribution, Tshadow_te_pre_y_distribution)
    print("==========================================================================================================================================")

    print("Training TS-ST attack model with 20000 train data and 20000 test data:")
    args.model = 'TS-ST_AM'
    attack_model(args, Starget_tr_pre_y_distribution, Starget_te_pre_y_distribution, Tshadow_tr_pre_y_distribution, Tshadow_te_pre_y_distribution)
    print("==========================================================================================================================================")

    print("Training SS-TT attack model with 20000 train data and 20000 test data:")
    args.model = 'SS-TT_AM'
    attack_model(args, Ttarget_tr_pre_y_distribution, Ttarget_te_pre_y_distribution, Sshadow_tr_pre_y_distribution, Sshadow_te_pre_y_distribution)
    print("==========================================================================================================================================")

    print("Training SS-ST attack model with 20000 train data and 20000 test data:")
    args.model = 'SS-ST_AM'
    attack_model(args, Starget_tr_pre_y_distribution, Starget_te_pre_y_distribution, Sshadow_tr_pre_y_distribution, Sshadow_te_pre_y_distribution)
    print("==========================================================================================================================================")

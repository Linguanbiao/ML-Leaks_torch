import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import numpy as np
import re

def Paint(train_acc, test_acc, train_loss, test_loss):
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
    # plt.savefig('./picture/1.png')

    #show函数展示出这个图，如果没有这行代码，则程序完成绘图，但看不到
    plt.show()
# #
# train_acc = [0.9, 0.93, 0.97]
# test_acc =  [0.93, 0.93, 0.967]
# train_loss = [1000,800,500]
# test_loss = [1100,650,300]
# Paint(train_acc, test_acc, train_loss, test_loss)
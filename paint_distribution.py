import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import seaborn as sns


def paint_histogram(data_train, data_test):
    matplotlib.rcParams['font.sans-serif'] = ['Tahoma']
    matplotlib.rcParams['axes.unicode_minus'] = False

    plt.figure(figsize=(12, 12), dpi=80)

    ax1 = plt.subplot(221)
    ax1.hist(data_train, histtype='stepfilled', alpha=0.3)
    ax1.set_xlabel('Conference')
    ax1.set_ylabel('Number')
    ax1.set_title('Train')
    ax2 = plt.subplot(222)
    ax2.hist(data_test, histtype='stepfilled', alpha=0.3)
    ax2.set_xlabel('Conference')
    ax2.set_ylabel('Number')
    ax2.set_title('Test')
    ax3 = plt.subplot(212)
    ax3.hist(data_train, histtype='stepfilled', alpha=0.3)
    ax3.hist(data_test, histtype='stepfilled', alpha=0.3)
    ax3.set_xlabel('Conference')
    ax3.set_ylabel('Number')
    ax3.set_title('Train&Test')

    plt.savefig('../DP_distribution/cifar100_500_200.png')
    plt.show()


def paint_scatter(data_train, data_test):
    # data_train = np.load('confidence/styleGAN2_train_10520/SGD/train_confidence_50.npy')
    # data_test = np.load('confidence/styleGAN2_train_10520/SGD/test_confidence_50.npy')
    plt.figure(figsize=(12, 12), dpi=80)
    ax1 = plt.axes(projection='3d')
    ax1.scatter3D(data_train[:, 0], data_train[:, 1], data_train[:, 2], c='g')
    ax1.scatter3D(data_test[:, 0], data_test[:, 1], data_test[:, 2], c='r')
    ax1.set_xlim(0.0, 1.0)
    ax1.set_title('Train&Test')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')

    plt.show()


if __name__ == '__main__':
    pass

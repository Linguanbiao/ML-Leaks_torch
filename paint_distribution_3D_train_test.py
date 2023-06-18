import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from mpl_toolkits.mplot3d import Axes3D


def paint_scatter():
    data_train = np.load('confidence/styleGAN2_train_10520/SGD/train_confidence_50.npy')
    data_test = np.load('confidence/styleGAN2_train_10520/SGD/test_confidence_50.npy')
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
    paint_scatter()

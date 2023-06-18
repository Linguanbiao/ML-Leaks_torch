import torch

from initialization import load_data
from sklearn.metrics import classification_report, accuracy_score


from data_partition import clipDataTopX
from train import iterate_minibatches


def test():
    # targetX, targetY = load_data("../dataDPSGD/dataCIFAR100/10000_250_200_0.2/targetModel.npz")
    targetX, targetY = load_data("../dataStyleGAN/dataCIFAR10_FID/10520/targetModel.npz")
    targetX = clipDataTopX(targetX, top=1)
    t =0.98



    print(t)
    pred_y = []
    for i in targetX:
        if i>t:
            pred_y.append(1)
        else:
            pred_y.append(0)

    print('test Accuracy: {}'.format(accuracy_score(targetY, pred_y)))
    print('More detailed results:')
    print(classification_report(targetY, pred_y))




if __name__ == '__main__':
    test()
import numpy as np
from paint import Paint
import lasagne
import theano
import theano.tensor as T
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import multilabel_confusion_matrix

def trainAttackModel(X_train, y_train, X_test, y_test):

    dataset = (X_train.astype(np.float32),
               y_train.astype(np.int32),
               X_test.astype(np.float32),
               y_test.astype(np.int32))

    output = train_model(dataset=dataset,
                            epochs=50,
                            batch_size=10,
                            learning_rate=0.03,
                            n_hidden=64,
                            l2_ratio=1e-6,
                            model='softmax')

    return output


def get_softmax_model(n_in, n_out):
    net = dict()
    net['input'] = lasagne.layers.InputLayer((None, n_in[1]))

    net['output'] = lasagne.layers.DenseLayer(
        net['input'],
        num_units=n_out,
        nonlinearity=lasagne.nonlinearities.softmax)
    return net


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


def clipDataTopX(dataToClip, top=3):
    res = [sorted(s, reverse=True)[0:top] for s in dataToClip]
    return np.array(res)


def train_model(dataset, n_hidden=50, batch_size=100, epochs=100, learning_rate=0.01, model='nn', l2_ratio=1e-7):

    train_x, train_y, test_x, test_y = dataset

    n_in = train_x.shape  # n_in (10520,3,32,32)  输入的shape

    n_out = len(np.unique(train_y))  # n_out 10   输出的shape

    if batch_size > len(train_y):
        batch_size = len(train_y)
    print('Building model with {} training data, {} classes...'.format(len(train_x), n_out))
    if model == 'cnn' or model == 'cnn2' or model == 'Droppcnn' or model == 'Droppcnn2':
        input_var = T.tensor4('x')  # 将x转换成tensor
    else:
        input_var = T.matrix('x')
    target_var = T.ivector('y')

    net = get_softmax_model(n_in, n_out)

    net['input'].input_var = input_var

    output_layer = net['output']  # (None,128)->(None,10) 模型输出层的参数
    # create loss function
    prediction = lasagne.layers.get_output(output_layer)
    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    loss = loss.mean() + l2_ratio * lasagne.regularization.regularize_network_params(output_layer,
                                                                                     lasagne.regularization.l2)
    # create parameter update expressions
    params = lasagne.layers.get_all_params(output_layer, trainable=True)  # params = [W, b, W, b, W, b, W, b]
    updates = lasagne.updates.adam(loss, params, learning_rate=learning_rate)
    train_fn = theano.function([input_var, target_var], loss, updates=updates)
    train_test_fn = theano.function([input_var, target_var], loss)
    # use trained network for predictions
    test_prediction = lasagne.layers.get_output(output_layer, deterministic=True)
    test_fn = theano.function([input_var], test_prediction)

    # 训练的同时测试
    accuracy = 0
    best_epoch = 0
    best_pre_y = []
    log_train_loss = []
    log_train_acc = []
    log_test_loss = []
    log_test_acc = []
    for epoch in range(epochs):
        # print('Training...')

        loss = 0
        pred_y = []
        train_pred_y = []
        for input_batch, target_batch in iterate_minibatches(train_x, train_y, batch_size):
            loss += train_fn(input_batch, target_batch)
            pred = test_fn(input_batch)
            pred_y.append(np.argmax(pred, axis=1))
            train_pred_y.append(target_batch)

        loss = round(loss, 3)
        pred_y = np.concatenate(pred_y)
        train_pred_y = np.concatenate(train_pred_y)
        # if(epoch % 10 ==0):
        print('Epoch {}, train loss {}, '.format(epoch, loss))
        log_train_loss.append(loss)
        t = accuracy_score(train_pred_y, pred_y)
        print('Training Accuracy: {}'.format(t))
        log_train_acc.append(t)

        t = 0
        if test_x is not None:
            print('Testing...')
            pred_y = []
            temp_loss = 0

            if batch_size > len(test_y):
                batch_size = len(test_y)

            for input_batch, target_batch in iterate_minibatches(test_x, test_y, batch_size, shuffle=False):
                # input_batch = (np.reshape(input_batch,(len(input_batch),3,32,32)))
                pred = test_fn(input_batch)
                loss = train_test_fn(input_batch, target_batch)
                #loss1 = train_fn(input_batch, target_batch)

                temp_loss += loss
                pred_y.append(np.argmax(pred, axis=1))

            temp_loss = round(temp_loss, 3)
            pred_y = np.concatenate(pred_y)
            t = accuracy_score(test_y, pred_y)
            print('Testing Accuracy: {}'.format(t))
            log_test_loss.append(temp_loss)
            log_test_acc.append(t)

        if (accuracy < t):
            accuracy = t
            best_epoch = epoch
            best_pre_y = pred_y


    print("最后一个epoch测试结果")
    print('More detailed results:')
    # print(test_y[],pred_y[5550:5650])
    print(classification_report(test_y, pred_y))

    if (len(np.unique(test_y)) == 10):
        mcm = multilabel_confusion_matrix(test_y, pred_y, labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        print(mcm)

    if (len(np.unique(test_y)) == 2):
        mcm = multilabel_confusion_matrix(test_y, pred_y, labels=[0, 1])
        print(mcm)

    print("===> When epoch == %d will get BEST ACC. PERFORMANCE: %.3f%%" % (best_epoch, accuracy))
    print('More detailed results:')
    print(classification_report(test_y, best_pre_y))

    if (len(np.unique(test_y)) == 10):
        mcm = multilabel_confusion_matrix(test_y, best_pre_y, labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        print(mcm)

    if (len(np.unique(test_y)) == 2):
        mcm = multilabel_confusion_matrix(test_y, best_pre_y, labels=[0, 1])
        print(mcm)

    Paint(log_train_acc, log_test_acc, log_train_loss, log_test_loss)

    return output_layer

def load_data(data_name):
    with np.load(data_name) as f:
        train_x, train_y = [f['arr_%d' % i] for i in range(len(f.files))]
    return train_x, train_y


# shadowX, shadowY = load_data("./dataAdult/5600/targetModelData.npz")
# targetX, targetY = load_data("./dataAdult/5600/targetModelData.npz")

# shadowX, shadowY = load_data("./dataNews/4500/shadowModelData.npz")
# targetX, targetY = load_data("./dataNews/4500/targetModelData.npz")


shadowX, shadowY = load_data("./dataCIFAR10/10520/shadowModelData.npz")
targetX, targetY = load_data("./dataCIFAR10/10520/targetModelData.npz")

# shadowX, shadowY = load_data("./dataMINST/10520/targetModelData.npz")
# targetX, targetY = load_data("./dataMINST/10520/targetModelData.npz")

# shadowX, shadowY = load_data("./dataCIFAR100/10520/targetModelData.npz")
# targetX, targetY = load_data("./dataCIFAR100/10520/targetModelData.npz")

# shadowX, shadowY = load_data("./dataLFW/420/targetModelData.npz")
# targetX, targetY = load_data("./dataLFW/420/targetModelData.npz")

# targetX = clipDataTopX(targetX, top=2)
# shadowX = clipDataTopX(shadowX, top=2)

print(shadowX.shape)
print(targetX.shape)



net = trainAttackModel(shadowX, shadowY, targetX, targetY)
modelPath = './model/CIFAR10'
np.savez(modelPath + '/attackModel.npz', * net.state_dict())
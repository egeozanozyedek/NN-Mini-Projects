"""
Author: Ege Ozan Özyedek
ID: 21703374
School: Bilkent University
Course: EEE443 - Neural Networks
"""


import numpy as np
import h5py
import matplotlib.pyplot as plt
import sys


class Neural_Network(object):
    """
    A neural network class which is used for both questions (Q1 and Q2). I decided to code a class for this assignment
    because this trivializes many things and is much more efficient since both questions have a lot in common. All
    functions in this class have documentation, and I have added as many inline comments as possible for throughout
    explanation of the class.
    """

    def __init__(self, size, layer_size = 2, question = 1, std = 0.01):
        """
        Initialization function for the Neural_Network class
        :param size: a length = layer_size+1 array which contains sizes of each layer, ex: size = [750 D P 250] for Q2
        :param layer_size: the layer size, ex: 1 for single-layer network (input is not counted as a layer)
        :param std: the standard deviation for weight initialization
        :param question: the question number, used in various places in the class to differentiate operations between
        the two questions that use this class
        """


        self.seed = 2222  # Choosing a seed reduces unpredictability for the testing of the code

        assert len(size) == layer_size + 1
        assert question == 1 or question == 2

        W = []
        b = []

        # in this for loop the weights and biases are initialized. For both questions I use N(0, 0.01).
        for i in range(layer_size):

            # We initialize the first weight matrix as [E E E], this is the embedding matrix. b is zero since
            # it has no use. This way I can update this matrix with regular gradient descent while also keeping
            # E the same. More info on this can be found in the report
            if question == 2 and i == 0:
                np.random.seed(self.seed)
                E = np.random.normal(0, std, size=(int(size[i]/3), size[i + 1]))
                W.append(np.vstack((E, E, E)))
                assert W[i].shape == (size[i], size[i + 1])
                np.random.seed(self.seed)
                b.append(np.zeros((1, size[i + 1])))
                continue
            np.random.seed(self.seed)
            W.append(np.random.normal(0, std, size=(size[i], size[i + 1])))
            np.random.seed(self.seed)
            b.append(np.random.normal(0, std, size=(1, size[i + 1])))
            assert W[i].shape == (size[i], size[i + 1])
            assert b[i].shape == (1, size[i + 1])

        # initialization of some class parameters
        self.momentum = {"W": [None] * layer_size, "b": [None] * layer_size}
        self.size = size
        self.params = {"W": W, "b": b}
        self.layer_size = layer_size
        self.q = question

        # setting the activation function sequence
        if question == 1:
            self.a = ["tanh"] * layer_size
        if question == 2:
            self.a = ["sigmoid"] * (layer_size - 1) + ["softmax"]

    def train(self, X, Y, X_val, Y_val, learning_rate = 0.5, epoch = 100, batch_size = 100, alpha = 0):
        """
        This funcion trains the neural network with given parameters.
        :param X: training input matrix
        :param Y: training label matrix
        :param X_val: validation input matrix
        :param Y_val: validation label matrix
        :param learning_rate: the learning rate for stochastic gradient descent
        :param epoch: iterations over the whole training set, epoch
        :param batch_size: the mini batch size which is used to find iterations per epoch
        :param alpha: the multiplier for the momentum term
        :return: A dictionary which contains all error metrics. The error is MSE for Q1 and cross entropy for Q2.
        """

        train_loss_list = []
        val_loss_list = []
        train_acc_list = []
        val_acc_list = []

        iter_per_epoch = int(X.shape[0] / batch_size)

        # question  2 requires the input to be one hot encoded to the given index, hence these are encoded
        if self.q == 2:
            Y_decoded = Y
            X = self.one_hot_encoder(X)
            Y = self.one_hot_encoder(Y)
            X_val = self.one_hot_encoder(X_val)
            Y_val_encoded = self.one_hot_encoder(Y_val)

        # training is done in this for loop
        for i in range(epoch):

            # shuffle
            p = self.shuffle(X.shape[0])
            X = X[p]
            Y = Y[p]
            if self.q == 2: Y_decoded = Y_decoded[p]

            # initialize the momentum terms to zero before every epoch
            for l in range(self.layer_size):
                self.momentum["W"][l] = np.zeros((self.size[l], self.size[l+1]))
                self.momentum["b"][l] = np.zeros((1, self.size[l+1]))

            # start and end indexes for mini  batches
            start = 0
            end = batch_size
            train_loss = 0

            # here the training over each mini-batch is done
            for j in range(iter_per_epoch):

                # choose mini-batch from the shuffled data
                X_mini = X[start:end]
                Y_mini = Y[start:end]

                if self.q == 2:
                    Y_decoded_mini = Y_decoded[start:end]

                loss, grads = self.loss(X_mini, Y_mini)
                train_loss += loss

                # gradient descent updates are done in this loop
                for k in range(self.layer_size):

                    if k == 0 and self.q == 2:  # For embedding matrix
                        self.momentum["W"][k] = learning_rate * grads["W"][k] + alpha * self.momentum["W"][k]
                        dE0, dE1, dE2 = np.array_split(self.momentum["W"][k], 3, axis=0)
                        dE = (dE0 + dE1 + dE2)/3
                        self.params["W"][k] -= np.vstack((dE, dE, dE))
                        assert self.params["W"][k].shape == (self.size[0], self.size[1])
                        continue

                    self.momentum["W"][k] = learning_rate * grads["W"][k] + alpha * self.momentum["W"][k]
                    self.momentum["b"][k] = learning_rate * grads["b"][k] + alpha * self.momentum["b"][k]
                    self.params["W"][k] -= self.momentum["W"][k]
                    self.params["b"][k] -= self.momentum["b"][k]

                # onto the next batch
                start = end
                end += batch_size


            # predictions
            if self.q == 1:
                # train_loss = self.MSE(Y, self.predict(X, False)) #FIXME
                val_loss = self.MSE(Y_val, self.predict(X_val, False))
                train_acc = (self.predict(X) == Y).mean() * 100
                val_acc = (self.predict(X_val) == Y_val).mean() * 100


            if self.q == 2:
                pred = self.predict(X)
                assert pred.shape == Y_decoded.shape
                train_acc = (pred == Y_decoded).mean() * 100 #FIXME

                pred = self.predict(X_val)
                assert pred.shape == Y_val.shape
                val_acc = (pred == Y_val).mean() * 100

                # train_pred = self.predict(X, False)
                # assert Y.shape == train_pred.shape
                # train_loss = self.cross_entropy(Y, train_pred) #FIXME

                val_pred = self.predict(X_val, False)
                assert Y_val_encoded.shape == val_pred.shape
                val_loss = self.cross_entropy(Y_val_encoded, val_pred)

            if self.q == 2:
                print('\r(D, P) = (%d, %d). tl: %f, vl: %f, ta: %f, va: %f [%d of %d].'
                      % (self.size[1], self.size[2], train_loss/(j+1), val_loss, train_acc, val_acc, i + 1, epoch), end='')

            train_loss_list.append(train_loss/iter_per_epoch)
            val_loss_list.append(val_loss)
            train_acc_list.append(train_acc)
            val_acc_list.append(val_acc)


            if i > 15 and self.q == 2:
                conv = val_loss_list[-15:]
                conv = sum(conv) / len(conv)

                limit = 0.05

                if (conv - limit) < val_loss < (conv + limit) and val_loss < 3.5:
                    print(" Training stopped since validation cross entropy reached convergence "
                          "(+- 0.05 from avg of past 15 data points).\n")
                    return {"train_loss_list": train_loss_list, "val_loss_list": val_loss_list,
                            "train_acc_list": train_acc_list, "val_acc_list": val_acc_list}


        return {"train_loss_list": train_loss_list, "val_loss_list": val_loss_list,
                "train_acc_list": train_acc_list, "val_acc_list": val_acc_list}

    def loss(self, X, Y):
        """
        This function computes the gradients and also the loss of given X (which is the training data)
        :param X: training data
        :param Y: train labels
        :return: loss and grads, the loss of X and the gradients for gradient descent updates
        """

        W = self.params["W"]
        b = self.params["b"]
        a = self.a
        out = [X]
        drv = [1]
        batch_size = X.shape[0]

        # forward propagation
        for i in range(self.layer_size):
            v = out[i] @ W[i] + b[i]
            o, d = self.activation(a[i], v)
            out.append(o)
            drv.append(d)

        # Compute the loss
        pred = out[-1]

        if self.q == 1:
            loss = self.MSE(Y, pred)
            delta = - (Y - pred) / batch_size * drv[-1]

        if self.q == 2:
            loss = self.cross_entropy(Y, pred)
            delta = pred
            delta[Y == 1] -= 1
            delta = delta / batch_size

        # Compute grads

        dW = []
        db = []
        ones = np.ones((1, batch_size))

        # backward propagation using delta
        for i in reversed(range(self.layer_size)):
            dW.append(out[i].T @ delta)
            db.append(ones @ delta)
            delta = drv[i] * (delta @ W[i].T)

        grads = {'W': dW[::-1], 'b': db[::-1]}

        return loss, grads

    def predict(self, X, classify=True):
        """
        This function predicts any input using the updated weights (the weights assigned to the network). Throughout the
        class this functuion is used to find the train and validation accuracies and the validation loss
        :param X: input for prediction
        :param classify: For both questions, to find accuracies we have to compare labels with classified (for the 1st
        question this means 1 or -1, for the 2nd question this is the max argument of the row). But we also need the
        non-classified "raw" prediction to find errors such as MSE and cross entropy. Hence by using this parameter
        we choose the output accordingly.
        :return: the prediction
        """
        W = self.params["W"]
        b = self.params["b"]
        a = self.a
        out = [X]

        # Prediction in this loop
        for i in range(self.layer_size):
            v = out[i] @ W[i] + b[i]
            out.append(self.activation(a[i], v)[0])

        # As a general note, out[-1] is the prediction. Its the final output of the forward propagation.
        if self.q == 1:
            return np.sign(out[-1]) if classify is True else out[-1]

        if self.q == 2:

            # This may seem a little random here but it has a purpose. This here asserts that the first weight matrix,
            # also known as the "embedding matrix" is equal for all 3 inputs. If this assertion was wrong it would mean
            # the code could not hold the assignment requirements. Thankfully, it works!
            E0, E1, E2 = np.array_split(self.params["W"][0], 3, axis=0)
            assert (E0 == E1).all() and (E1 == E2).all()

            # This is for the classified input, we find the max arguments over each row and add 1.
            #  We add one here because while encoding the input arrays we deduce one to correctly encode, hence the
            #  below code finds arguments from 0 to 249. However, label elements start at 1 upto 250. Hence we add one
            #  to correctly find the classification accuracy.
            o = (np.argmax(out[-1], axis=1) + 1).T
            o = np.reshape(o, (o.shape[0], 1))
            return o if classify is True else out[-1]

    def one_hot_encoder(self, X, size=250):
        """
        One hot encoder, ex.: 144 -> [0 ... 1 ... 0] where 1 is in the index 143
        :param X: input data
        :param size: 250 in our case, this would be the maximum index+1 for other examples
        :return: encoded data
        """
        X = X - 1
        encodedX = np.zeros((X.shape[0], 0))

        for i in range(X.shape[1]):
            temp = np.zeros((X.shape[0], size))
            temp[np.arange(X.shape[0]), X[:, i]] = 1
            encodedX = np.hstack((encodedX, temp))

        return encodedX

    def cross_entropy(self, desired, output):
        """
        This function finds the cross entropy error
        :param desired: desired point, label data
        :param output: output, prediction
        :return: the cross entropy error
        """
        assert len(desired) == len(output)
        return np.sum(- desired * np.log(output)) / desired.shape[0]

    def activation(self, a, X):
        """
        This function finds activation values and their corresponding derivatives at that point
        :param a: the activation sequence of the network, this is determined at the initialization for the network
        :param X: the input at each step of forward propagation; u, h, ...
        :return: the output of the activation function and its corresponding derivative value. One exception is
        the softmax function in which the derivative is passed as None. This is because for Q2 we find the delta
        (which is derv of loss * derv of softmax) directly and dont have to explicitly write out the derivative of the
        softmax function.
        """
        if a == "tanh":
            activation = np.tanh(X)
            derivative = 1 - activation**2
            return activation, derivative
        if a == "sigmoid":
            activation = 1 / (1 + np.exp(-X))
            derivative = activation * (1 - activation)
            return activation, derivative
        if a == "softmax":
            activation = np.exp(X) / np.sum(np.exp(X), axis= 1, keepdims=True)
            derivative = None
            return activation, derivative
        return None

    def MSE(self, desired, output):
        assert len(desired) == len(output)
        return ((desired - output)**2).mean()

    def shuffle(self, length):
        """
        This function outputs a permutation, which is used for shuffling arrays
        :param length: the length of the permutation, in our case this is the batch size
        :return: the permutation
        """
        np.random.seed(self.seed)
        p = np.random.permutation(length)
        return p


def question_1():
    """
    The first question of the assignment.
    """
    print("\nDisclaimer: This question will save images of the plots into the file directory its in.\n\n")

    ##############################
    # ACQUIRE DATA
    ##############################

    filename = "assign2_data1.h5"
    h5 = h5py.File(filename, 'r')
    trainims = h5['trainims'][()].astype('float64').transpose(0, 2, 1)
    trainlbls = h5['trainlbls'][()].astype(int)
    testims = h5['testims'][()].astype('float64').transpose(0, 2, 1)
    testlbls = h5['testlbls'][()].astype(int)
    h5.close()

    trainlbls[np.where(trainlbls == 0)] = -1
    testlbls[np.where(testlbls == 0)] = -1

    X = np.reshape(trainims, (trainims.shape[0], trainims.shape[1] * trainims.shape[2]))
    X = 1 * X / np.amax(X)
    Y = np.reshape(trainlbls, (trainlbls.shape[0], 1))
    X_test = np.reshape(testims, (testims.shape[0], testims.shape[1] * testims.shape[2]))
    X_test = 1 * X_test / np.amax(X_test)
    Y_test = np.reshape(testlbls, (testlbls.shape[0], 1))

    ############################
    # SETUP
    ############################

    learning_rate = 0.25
    hidden_layer = [20, 2, 200]
    batch_size = 50
    epoch = 300
    alpha = 0
    last = epoch - 50
    size = [X.shape[1], hidden_layer[0], 1]
    layer_size = 2

    mse_test = []
    mse_train = []
    acc_test = []
    acc_train = []

    ############################
    # QUESTION 1.a
    ############################

    network = Neural_Network(size, layer_size, question=1)
    info = network.train(X, Y, X_test, Y_test, learning_rate, epoch, batch_size, alpha=alpha)
    mse, mce, train_acc, test_acc = info.values()

    msea = mse
    mcea = mce
    train_acca = train_acc
    test_acca = test_acc

    fig = plt.figure(figsize=(20, 15), dpi=80, facecolor='w', edgecolor='k')
    fig.suptitle("Question 1.a - All Error Metrics for \u03B7=" + str(learning_rate) + ", hidden neurons=" + str(
        hidden_layer[0]) + ", batch size=" + str(batch_size), fontsize=20)
    plt.subplot(2, 2, 1)
    plt.plot(mse, "C3")
    plt.title("MSE for Train Data")
    plt.xlabel("Epoch")
    plt.subplot(2, 2, 2)
    plt.plot(mce, "C2")
    plt.title("MSE for Test Data")
    plt.xlabel("Epoch")
    plt.subplot(2, 2, 3)
    plt.plot(train_acc, "C3")
    plt.title("Train Data Prediction Accuracy (1 - Classification Error)")
    plt.xlabel("Epoch")
    plt.subplot(2, 2, 4)
    plt.plot(test_acc, "C2")
    plt.title("Test Data Prediction Accuracy (1 - Classification Error)")
    plt.xlabel("Epoch")

    plt.savefig("q1a.png")

    print("-----------[learning rate = " + str(learning_rate) + "]---[hidden size = " + str(
        hidden_layer) + "]---[batch_size = " + str(batch_size) + "]-----------\n")
    print("Avg of last " + str(last) + " Mean Squared Error", sum(mse[-last:]) / last)
    print("Avg of last " + str(last) + " Mean Classification Error", sum(mce[-last:]) / last)
    print("Avg of last " + str(last) + " Train Accuracies", sum(train_acc[-last:]) / last)
    print("Avg of last " + str(last) + " Test Accuracies", sum(test_acc[-last:]) / last, "\n")
    print("----------------------------------------------------------------------------------\n")

    ############################
    # QUESTION 1.c
    ############################

    for h in hidden_layer:
        size = [X.shape[1], h, 1]
        network = Neural_Network(size, layer_size, question=1)
        info = network.train(X, Y, X_test, Y_test, learning_rate, epoch, batch_size, alpha=alpha)
        mse, mce, train_acc, test_acc = info.values()
        mse_train.append(mse)
        mse_test.append(mce)
        acc_train.append(train_acc)
        acc_test.append(test_acc)

    fig = plt.figure(figsize=(20, 15), dpi=80, facecolor='w', edgecolor='k')
    fig.suptitle('Question 1.c - NN with Different Hidden Neuron Amounts', fontsize=20)
    plt.subplot(2, 2, 1)

    plt.plot(mse_train[1], "C2", label="N = 2")
    plt.plot(mse_train[2], "C3", label="N = 200")
    plt.plot(mse_train[0], "C1", label="N = 20")
    plt.legend()
    plt.title("MSE for Train Data")
    plt.xlabel("Epoch")
    plt.subplot(2, 2, 2)

    plt.plot(mse_test[1], "C2", label="N = 2")
    plt.plot(mse_test[2], "C3", label="N = 200")
    plt.plot(mse_test[0], "C1", label="N = 20")
    plt.legend()
    plt.title("MSE for Test Data")
    plt.xlabel("Epoch")
    plt.subplot(2, 2, 3)

    plt.plot(acc_train[1], "C2", label="N = 2")
    plt.plot(acc_train[2], "C3", label="N = 200")
    plt.plot(acc_train[0], "C1", label="N = 20")
    plt.legend()
    plt.title("Train Data Prediction Accuracy (1 - Classification Error)")
    plt.xlabel("Epoch")
    plt.subplot(2, 2, 4)

    plt.plot(acc_test[1], "C2", label="N = 2")
    plt.plot(acc_test[2], "C3", label="N = 200")
    plt.plot(acc_test[0], "C1", label="N = 20")
    plt.legend()
    plt.title("Test Data Prediction Accuracy (1 - Classification Error)")
    plt.xlabel("Epoch")

    plt.savefig("q1c.png")

    ############################
    # QUESTION 1.d
    ############################

    learning_rate = 0.1
    hidden_layer = [20, 15]
    batch_size = 50

    size = [X.shape[1]]
    size += hidden_layer
    size.append(1)

    layer_size = len(hidden_layer) + 1

    network = Neural_Network(size, layer_size, question=1)
    info = network.train(X, Y, X_test, Y_test, learning_rate, epoch, batch_size, alpha=0)
    mse, mce, train_acc, test_acc = info.values()

    print("-----------[learning rate = " + str(learning_rate) + "]---[hidden size = " + str(
        hidden_layer) + "]---[batch_size = " + str(batch_size) + "]-----------\n")
    print("Avg of last " + str(last) + " Mean Squared Error", sum(mse[-last:]) / last)
    print("Avg of last " + str(last) + " Mean Classification Error", sum(mce[-last:]) / last)
    print("Avg of last " + str(last) + " Train Accuracies", sum(train_acc[-last:]) / last)
    print("Avg of last " + str(last) + " Test Accuracies", sum(test_acc[-last:]) / last, "\n")
    print("----------------------------------------------------------------------------------\n")

    fig = plt.figure(figsize=(20, 15), dpi=80, facecolor='w', edgecolor='k')
    fig.suptitle("Question 1.d - NN with 2 Hidden Layers with \u03B7=" + str(learning_rate) + ", hidden neurons=" + str(
        hidden_layer) + ", batch size=" + str(batch_size), fontsize=20)
    plt.subplot(2, 2, 1)
    plt.plot(mse, "C3")
    plt.title("MSE for Train Data")
    plt.xlabel("Epoch")
    plt.subplot(2, 2, 2)
    plt.plot(mce, "C2")
    plt.title("MSE for Test Data")
    plt.xlabel("Epoch")
    plt.subplot(2, 2, 3)
    plt.plot(train_acc, "C3")
    plt.title("Train Data Prediction Accuracy (1 - Classification Error)")
    plt.xlabel("Epoch")
    plt.subplot(2, 2, 4)
    plt.plot(test_acc, "C2")
    plt.title("Test Data Prediction Accuracy (1 - Classification Error)")
    plt.xlabel("Epoch")

    plt.savefig("q1d.png")

    fig = plt.figure(figsize=(20, 15), dpi=80, facecolor='w', edgecolor='k')
    fig.suptitle("Question 1.d - Comparison between NN's with different layer sizes", fontsize=20)
    plt.subplot(2, 2, 1)
    plt.plot(msea, "C1", label="layer size = 1")
    plt.plot(mse, "C4", label="layer size = 2")
    plt.legend()
    plt.title("MSE for Train Data")
    plt.xlabel("Epoch")
    plt.subplot(2, 2, 2)
    plt.plot(mcea, "C1", label="layer size = 1")
    plt.plot(mce, "C4", label="layer size = 2")
    plt.legend()
    plt.title("MSE for Test Data")
    plt.xlabel("Epoch")
    plt.subplot(2, 2, 3)
    plt.plot(train_acca, "C1", label="layer size = 1")
    plt.plot(train_acc, "C4", label="layer size = 2")
    plt.legend()
    plt.title("Train Data Prediction Accuracy (1 - Classification Error)")
    plt.xlabel("Epoch")
    plt.subplot(2, 2, 4)
    plt.plot(test_acca, "C1", label="layer size = 1")
    plt.plot(test_acc, "C4", label="layer size = 2")
    plt.legend()
    plt.title("Test Data Prediction Accuracy (1 - Classification Error)")
    plt.xlabel("Epoch")

    plt.savefig("q1d-alt.png")

    mse1 = mse
    mce1 = mce
    ta = train_acc
    tea = test_acc

    ############################
    # QUESTION 1.e
    ############################

    alpha = 0.5

    network = Neural_Network(size, layer_size, question=1)
    info = network.train(X, Y, X_test, Y_test, learning_rate, epoch, batch_size, alpha=alpha)
    mse, mce, train_acc, test_acc = info.values()

    print("-----------[learning rate = " + str(learning_rate) + "]---[hidden size = " + str(
        hidden_layer) + "]---[batch_size = " + str(batch_size) + "]---[alpha = " + str(alpha) + "]-----------\n")
    print("Avg of last " + str(last) + " Mean Squared Error", sum(mse[-last:]) / last)
    print("Avg of last " + str(last) + " Mean Classification Error", sum(mce[-last:]) / last)
    print("Avg of last " + str(last) + " Train Accuracies", sum(train_acc[-last:]) / last)
    print("Avg of last " + str(last) + " Test Accuracies", sum(test_acc[-last:]) / last, "\n")
    print("----------------------------------------------------------------------------------\n")

    fig = plt.figure(figsize=(20, 15), dpi=80, facecolor='w', edgecolor='k')
    fig.suptitle('Question 1.e - NN with 2 Hidden Layers and Momentum Coefficient \u03B1 =' + str(alpha), fontsize=20)
    plt.subplot(2, 2, 1)
    plt.plot(mse, "C3")
    plt.title("MSE for Train Data")
    plt.xlabel("Epoch")
    plt.subplot(2, 2, 2)
    plt.plot(mce, "C2")
    plt.title("MSE for Test Data")
    plt.xlabel("Epoch")
    plt.subplot(2, 2, 3)
    plt.plot(train_acc, "C3")
    plt.title("Train Data Prediction Accuracy (1 - Classification Error)")
    plt.xlabel("Epoch")
    plt.subplot(2, 2, 4)
    plt.plot(test_acc, "C2")
    plt.title("Test Data Prediction Accuracy (1 - Classification Error)")
    plt.xlabel("Epoch")

    plt.savefig("q1e.png")

    fig = plt.figure(figsize=(20, 15), dpi=80, facecolor='w', edgecolor='k')
    fig.suptitle("Question 1.e - Comparison between NN's with vs without momentum term", fontsize=20)
    plt.subplot(2, 2, 1)
    plt.plot(mse1, "C1", label="\u03B1 = 0")
    plt.plot(mse, "C4", label="\u03B1 = " + str(alpha))
    plt.legend()
    plt.title("MSE for Train Data")
    plt.xlabel("Epoch")
    plt.subplot(2, 2, 2)
    plt.plot(mce1, "C1", label="\u03B1 = 0")
    plt.plot(mce, "C4", label="\u03B1 = " + str(alpha))
    plt.legend()
    plt.title("MSE for Test Data")
    plt.xlabel("Epoch")
    plt.subplot(2, 2, 3)
    plt.plot(ta, "C1", label="\u03B1 = 0")
    plt.plot(train_acc, "C4", label="\u03B1 = " + str(alpha))
    plt.legend()
    plt.title("Train Data Prediction Accuracy (1 - Classification Error)")
    plt.xlabel("Epoch")
    plt.subplot(2, 2, 4)
    plt.plot(tea, "C1", label="\u03B1 = 0")
    plt.plot(test_acc, "C4", label="\u03B1 = " + str(alpha))
    plt.legend()
    plt.title("Test Data Prediction Accuracy (1 - Classification Error)")
    plt.xlabel("Epoch")

    plt.savefig("q1e-alt.png")


def question_2():
    """
    The second question of the assignment.
    """

    print("\nDisclaimer: This question will save images of the plots into the file directory its in."
          "\nNote: To reduce run time for testing, the epoch number is reduced to 5 from 50. "
          "The plots used in the report show the whole 50 epoch run.\n")

    ##############################
    # ACQUIRE DATA
    ##############################

    filename = "assign2_data2.h5"
    h5 = h5py.File(filename, 'r')
    words = h5['words'][()]
    trainx = h5['trainx'][()]
    traind = h5['traind'][()]
    valx = h5['valx'][()]
    vald = h5['vald'][()]
    testx = h5['testx'][()]
    testd = h5['testd'][()]
    h5.close()

    traind = np.reshape(traind, (traind.shape[0], 1))
    vald = np.reshape(vald, (vald.shape[0], 1))
    testd = np.reshape(testd, (testd.shape[0], 1))
    words = np.reshape(words, (words.shape[0], 1))

    ############################
    # QUESTION 2.a
    ############################

    learning_rate = 0.15
    alpha = 0.85
    epoch = 5  # to see full plots, change the epoch to 50
    batch_size = 200

    # 8 , 64
    ######################
    D = 8
    P = 64
    hidden_layer = [D, P]

    size = [750]
    size += hidden_layer
    size.append(250)

    layer_size = len(size) - 1


    fig = plt.figure(figsize=(30, 10), dpi=80, facecolor='w', edgecolor='k')
    fig.suptitle("Question 2.a - Train and Validation", fontsize=20)

    network = Neural_Network(size, layer_size, question=2)
    info = network.train(trainx, traind, valx, vald, learning_rate, epoch, batch_size, alpha)
    train_loss_list, val_loss_list, train_acc_list, val_acc_list = info.values()
    print()

    plt.subplot(1, 3, 1)
    plt.plot(train_loss_list, "C0", label="train loss")
    plt.plot(val_loss_list, "C3", label="val loss")
    plt.legend()
    plt.title("(D, P) = 8, 64")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")


    # 16, 128
    ######################

    D = 16
    P = 128

    hidden_layer = [D, P]

    size = [750]
    size += hidden_layer
    size.append(250)

    network = Neural_Network(size, layer_size, question=2)
    info = network.train(trainx, traind, valx, vald, learning_rate, epoch, batch_size, alpha)
    train_loss_list, val_loss_list, train_acc_list, val_acc_list = info.values()
    print()

    plt.subplot(1, 3, 2)
    plt.plot(train_loss_list, "C0", label="train loss")
    plt.plot(val_loss_list, "C3", label="val loss")
    plt.legend()
    plt.title("(D, P) = 16, 128")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")


    # 32, 256
    ######################

    D = 32
    P = 256

    hidden_layer = [D, P]

    size = [750]
    size += hidden_layer
    size.append(250)

    network = Neural_Network(size, layer_size, question=2)
    info = network.train(trainx, traind, valx, vald, learning_rate, epoch, batch_size, alpha)
    train_loss_list, val_loss_list, train_acc_list, val_acc_list = info.values()
    print()

    plt.subplot(1, 3, 3)
    plt.plot(train_loss_list, "C0", label="train loss")
    plt.plot(val_loss_list, "C3", label="val loss")
    plt.legend()
    plt.title("(D, P) = 32, 256")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")

    plt.savefig("q2a.png")

    ############################
    # QUESTION 2.b
    ############################

    test_pred_classified = network.predict(network.one_hot_encoder(testx))

    print("\n\nTest accuracy for (D, P) = (32, 256): ", (test_pred_classified == testd).mean() * 100)

    w = 10  # output 10 predictions

    p = network.shuffle(testx.shape[0])  # shuffle testd to chose randomly
    testx = testx[p][:w]
    testd = testd[p][:w]

    testx_e = network.one_hot_encoder(testx)
    test_pred = network.predict(testx_e, False)

    n = 10
    s = (np.argsort(-test_pred, axis=1) + 1)[:, :n]

    for i in range(w):
            print("\n--------------")
            print("Sentence:", words[testx[i][0] - 1], words[testx[i][1] - 1], words[testx[i][2] - 1])
            print("Label:", words[testd[i] - 1])
            for j in range(n):
                print(str(j + 1) + ". ", words[s[i][j] - 1])


def ege_ozan_ozyedek_21703374_hw2(question):
    if question == '1':
        question_1()
    elif question == '2':
        question_2()


# Ask the number of a question
question = sys.argv[1]
ege_ozan_ozyedek_21703374_hw2(question)

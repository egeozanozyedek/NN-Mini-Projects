import numpy as np
import h5py
import matplotlib.animation as animation
import matplotlib.pyplot as plt 
import string 
import sys
import math


#####################################################
#####################################################
# QUESTION 2   QUESTION 2   QUESTION 2   QUESTION 2
#####################################################
#####################################################


def question_2():
    X = np.array([
                [0, 0, 0, 0, -1],
                [0,0,0,1,-1],
                [0,0,1,0,-1],
                [0,0,1,1,-1],
                [0,1,0,0,-1],
                [0,1,0,1,-1],
                [0,1,1,0,-1],
                [0,1,1,1,-1],
                [1,0,0,0,-1],
                [1,0,0,1,-1],
                [1,0,1,0,-1],
                [1,0,1,1,-1],
                [1,1,0,0,-1],
                [1,1,0,1,-1],
                [1,1,1,0,-1],
                [1,1,1,1,-1]
                ])

    X = X.T
    X = X.astype(np.float64)

    o = np.array([
                [0],
                [0],
                [0],
                [1],
                [1],
                [1],
                [1],
                [0],
                [0],
                [0],
                [0],
                [1],
                [0],
                [0],
                [0],
                [1]
                ])
    
    h = np.array([
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 1.0, 0, 0, 0],
            [0, 0, 1.0, 1.0, 0],
            [0, 0, 1.0, 0, 0],
            [0, 0, 0, 1.0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [1.0, 1.0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [1.0, 0, 0, 0, 0],
            ])

    W_in = np.array([
                      [0.4, 0, 0.4, 0.4, 1],
                      [0, -0.6, 0.6, 0.6, 1],
                      [-0.4, 1.2, -0.4, 0, 1],
                      [-0.4, 1.2, 0, -0.4, 1]
                      ])
    

    W_h = np.array([
                    [1],
                    [1],
                    [1], 
                    [1],
                    [0.5]
                       ])
    
    W_in_up = np.array([
                    [1, 0, 1, 1, 2.5],
                    [0, -1, 1, 1, 1.5],
                    [-1, 1, -1, 0, 0.5],
                    [-1, 1, 0, -1, 0.5]
    ])

    trial = 1000
    result = 0
    result_1 = 0
    result_2 = 0

    for i in range(trial):
        result += question_2_accuracy(X,W_in,W_h,o);
        noise = np.random.normal(0, 0.1, size = (5, 16))
        result_1 += question_2_accuracy( (X + noise) , W_in, W_h, o)
        result_2 += question_2_accuracy( (X + noise), W_in_up, W_h, o)

    result /= trial
    result_1 /= trial
    result_2 /= trial

    print("Question 2.b\nAccuracy (mean of 100 trials): ", result, "%")
    print("Question 2.c\nAccuracy w/ noise applied to inputs (mean of 100 trials) ", result_1, "%")
    print("Question 2.c\nAccuracy after weights changed (mean of 100 trials):", result_2, "%")


    ## 2.d

    X = np.array([[0,0,0,0],
                [0,0,0,1],
                [0,0,1,0],
                [0,0,1,1],
                [0,1,0,0],
                [0,1,0,1],
                [0,1,1,0],
                [0,1,1,1],
                [1,0,0,0],
                [1,0,0,1],
                [1,0,1,0],
                [1,0,1,1],
                [1,1,0,0],
                [1,1,0,1],
                [1,1,1,0],
                [1,1,1,1]])
    
    d = np.array([
            [0],
            [0],
            [0],
            [1],
            [1],
            [1],
            [1],
            [0],
            [0],
            [0],
            [0],
            [1],
            [0],
            [0],
            [0],
            [1]
            ])
    
    X = X.astype(np.float64)      
    X = np.tile(X, (25,1))
    d = np.tile(d, (25,1))
    theta = np.full((400,1), -1)
    for i in range(100):
        noise = np.random.normal(0, 0.2, size = (X.shape[0], X.shape[1]))
        X_n = np.c_[X + noise, theta]
        result_1 += question_2_accuracy(X_n.T , W_in, W_h, d)
        result_2 += question_2_accuracy(X_n.T, W_in_up, W_h, d)

    print("Question 2.d\nAccuracy of 2.a (mean of 100 trials): ", result_1/100, "%")
    print("Accuracy of 2.c (mean of 100 trials)", result_2/100, "%")



def question_2_accuracy(X, W_in, W_h, d, update = False):
    temp = W_in @ X
    temp = temp.T
    assert(temp.shape == (d.shape[0],4))
    h = np.c_[temp, np.full((d.shape[0],1), -1)]
    assert(h.shape == (d.shape[0],5))
    h = np.heaviside(h, 0)
    o = h @ W_h
    o = np.heaviside(o, 0)
    return (d == o).mean() * 100

#####################################################
#####################################################
# QUESTION 3   QUESTION 3   QUESTION 3   QUESTION 3
#####################################################
#####################################################


def sigmoid(v ,lam = 1, polarity = 0):
    if polarity == 0:
        return 1 / (1 + np.exp(- v * lam))
    else:
        return (1 - np.exp(- v * lam)) / (1 + np.exp(- v * lam))


def corr_plot(ims):

    #flattens images
    ims = np.reshape(ims, (ims.shape[0], ims.shape[1] * ims.shape[2]))
    #chose 26 images, first of their class
    corr_ims = ims[0:-1:200]
    corr = np.corrcoef(corr_ims)

    f, axarr = plt.subplots(figsize=(15, 15), dpi=80)
    im = axarr.imshow(corr)

    for i in range(26):
        for j in range(26):
            text = axarr.text(j, i, float(np.round(corr[i,j], 2)),
                            ha="center", va="center", color="w")

    axarr.set_title("Correlation Coefficient Matrix between 26 images (one from each class)")
    Classes = list(string.ascii_lowercase)
    plt.xticks(range(26), Classes)
    plt.yticks(range(26), Classes)
    plt.colorbar(im);
    plt.show()


def disp(d = None, w = False, mse = False, mse_lol = None, learning_rate = None):
    if mse == True:

        plt.plot(mse_lol[1], 'r', label=('\u03B7 =' + str(learning_rate[1])))
        plt.plot(mse_lol[2], 'b', label=('\u03B7 =' + str(learning_rate[2])))
        plt.plot(mse_lol[0], 'g', label=('\u03B7 =' + str(learning_rate[0])))
        plt.legend(loc='best')
        plt.xlabel('Sample Size')
        plt.ylabel('MSE')
        plt.title("MSE for different \u03B7 values")
        plt.show()

        plt.figure()
        f, axarr = plt.subplots(3, figsize=(10,7)) 
        f.suptitle("MSE for different \u03B7 values")
        axarr[0].plot(mse_lol[0], 'g')
        axarr[0].set_title('\u03B7 =' + str(learning_rate[0]))
        axarr[1].plot(mse_lol[1], 'r')
        axarr[1].set_title('\u03B7 =' + str(learning_rate[1]))
        axarr[2].plot(mse_lol[2], 'b')
        axarr[2].set_title('\u03B7 =' + str(learning_rate[2]))
        plt.show()

    else:
        f, axarr = plt.subplots(5,6, figsize=(10,12)) 
        
        k = 0

        for i in range(5):
            for j in range(6):
                if k < 26:
                    if w == True:
                        axarr[i][j].imshow(d[k])
                        title = "WEIGHT_VISUAL.png"
                    else:
                        axarr[i][j].imshow(d[200*k].T)
                        title = "SAMPLE_VISUAL.png"
                    axarr[i][j].set_title("Class: " + str(k + 1))
                axarr[i][j].axis('off')
                k += 1
        f.suptitle(title)
        plt.show()



def question_3_init(ims_shape):

    mse_list = []

    W = np.random.normal(0, 0.01, size = (ims_shape[1] * ims_shape[2], 26))
    b = np.random.normal(0, 0.01, size = (1,1))
    o = np.zeros((26,1))
    dw = np.zeros((ims_shape[1] * ims_shape[2], 26))
    db = np.zeros((1,1))
    
    return W, b, o, dw, db, mse_list


def question_3_propagate(ims, lbls, W, b, o, dw, db, mse_list, learning_rate = 0.5, epoch = 10000, lam = 1):
    
    for i in range(epoch):

        k = np.random.randint(0, ims.shape[0])
  
        X = np.reshape(ims[k].T, (ims.shape[1] * ims.shape[2], 1))
        X = question_3_normalize(X)

        d = np.full((26,1), 0)
        d[int(lbls[k]) - 1] = 1

        o = sigmoid(W.T @ X - b, lam, 0)

        mse = np.sum((o - d) ** 2)/26
        mse_list.append(mse)

        ds =  (d - o) * lam * o * (1 - o)
        # ds =  (d - o) * (lam / 2.0) * (1 - o * o) #bipolar sigmoid 

        dW = learning_rate * ( X @ ds.T)
        db = learning_rate * ds * -1
        
        W = W + dW
        b = b + db

    return mse_list, W, b


def question_3_predict(ims, d, W, b, lam):
    ims = question_3_normalize(ims)
    o = sigmoid(ims @ W - b.T, lam)
    return (np.argmax(o, axis=1) == (d - 1)).mean() * 100



def question_3_normalize(ims):
    if ims.shape[0] > 1 and ims.shape[1] > 1:
        n = ims.max(axis = 1)
        ims *= 1.0/np.reshape(n, (-1,1))
    else:
        ims *= 1.0/ims.max()
    return ims



def question_3(filename = "assign1_data1.h5"):


    h5 = h5py.File(filename,'r')
    #train data
    trainims = h5['trainims'][()].astype('float64')
    trainlbls = h5['trainlbls'][()].astype('float64')
    #test data
    testims = h5['testims'][()].astype('float64')
    testlbls = h5['testlbls'][()].astype('float64')
    h5.close()

    #display 26 samples and their correlation coefficient matrix
    disp(trainims)
    corr_plot(trainims.transpose(0,2,1))

    mse_lol = [] #list of lists

    ###TRAINING###

    #determine variables
    ims = trainims
    lbls = trainlbls    
    epoch = 10000
    learning_rate = [0.146, 10, 0.0001]
    lam = 1
    a_list = []

    for eta in learning_rate:
        #initialize
        W, b, o, dw, db, mse_list = question_3_init(ims.shape)

        #train (both forward and back propagation)
        mse_list, W, b = question_3_propagate(ims, lbls, W, b, o, dw, db, mse_list, eta, epoch, lam)
        mse_lol.append(mse_list)

        #display the final weights, should resemble the actual class shapes
        if eta == learning_rate[0]:
            disp( np.reshape( W.T, (-1, ims.shape[1], ims.shape[2])) , True)

        #TEST: accuracy calculations

        # ##TRAIN ACCURACY
        t = trainims.transpose(0, 2, 1)
        t = np.reshape(t, (t.shape[0], t.shape[1] * t.shape[2]))
        tl = trainlbls    

        a = question_3_predict(t, tl, W, b, lam)
        
        
        # #TEST ACCURACY
        t = testims.transpose(0, 2, 1)
        t = np.reshape(t, (t.shape[0], t.shape[1] * t.shape[2]))
        tl = testlbls    
        b = question_3_predict(t, tl, W, b, lam)
        # a_list.append(a)
            
        print("Results for \u03B7 = " + str(eta))
        print("Train accuracy: ", a, " %")
        print("Test accuracy: ", b, " %")


    plt.figure(figsize=(10,5))
    disp(mse = True, mse_lol = mse_lol, learning_rate = learning_rate)


def ege_ozan_ozyedek_21703374_hw1(question):
    if question == '2' :
        question_2()
        ##question 2 code goes here
    elif question == '3' :
        question_3("assign1_data1.h5")##### PLEASE ENTER YOUR DATASET DIRECTORY HERE ####
        ##question 3 code goes here

ege_ozan_ozyedek_21703374_hw1('2')
ege_ozan_ozyedek_21703374_hw1('3')


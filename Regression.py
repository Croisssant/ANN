import numpy as np
import tensorflow as tf
import math
import logging
#logging.basicConfig(level=logging.DEBUG)
import matplotlib.pyplot as plt
import pandas as pd
import os

# Writing data into attribute files
def data_loader(dataset_path):
    df = pd.read_table(dataset_path, header=None)
    X = df.iloc[:, 0:-1]
    y = df.iloc[:, -1]
    data_folder = './batch_data/regression/'

    if os.path.exists(data_folder):
        return data_folder

    m = X.shape[0] # number of instances
    n = X.shape[1] # number of attributes

    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    X = (X-mu)/sigma

    if not os.path.exists(data_folder): 
        os.makedirs(data_folder)

    for i in range(n):
        X[i].to_csv('{}/x{}.txt'.format(data_folder, i+1), header=False, index=False)

    y.to_csv('{}/y1.txt'.format(data_folder), header=False, index=False)

    return data_folder

#Network parameters
n_hidden1 = 10
n_hidden2 = 10
n_input = 5
n_output = 1

#Learning parameters
learning_constant = 0.015
number_epochs = 5500
batch_size = 1150

#Defining the input and the output
X = tf.placeholder("float", [None, n_input])
Y = tf.placeholder("float", [None, n_output])

#DEFINING WEIGHTS AND BIASES

#Biases first hidden layer
b1 = tf.Variable(tf.random_normal([n_hidden1]))
#Biases second hidden layer
b2 = tf.Variable(tf.random_normal([n_hidden2]))
#Biases output layer
b3 = tf.Variable(tf.random_normal([n_output]))

#Weights connecting input layer with first hidden layer
w1 = tf.Variable(tf.random_normal([n_input, n_hidden1]))
#Weights connecting first hidden layer with second hidden layer
w2 = tf.Variable(tf.random_normal([n_hidden1, n_hidden2]))
#Weights connecting second hidden layer with output layer
w3 = tf.Variable(tf.random_normal([n_hidden2, n_output]))

def multilayer_perceptron(input_d):
    #Task of neurons of first hidden layer
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(input_d, w1), b1))
    #Task of neurons of second hidden layer
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, w2), b2))
    #Task of neurons of output layer
    out_layer = tf.add(tf.matmul(layer_2, w3), b3) 

    return out_layer

#Create model
neural_network = multilayer_perceptron(X)

#Define loss and optimizer
loss_op = tf.reduce_mean(tf.math.squared_difference(Y,neural_network))
#loss_op = tf.reduce_mean(abs(Y - neural_network))
#loss_op = tf.reduce_mean(100 * abs(Y - neural_network) / Y)
#loss_op = tf.reduce_mean(square(log(y_true + 1.) - log(y_pred + 1.)))
#oss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=neural_network,labels=Y))
optimizer = tf.train.GradientDescentOptimizer(learning_constant).minimize(loss_op)

#Initializing the variables
init = tf.global_variables_initializer()

data_folder = data_loader('./Data/Airfoil_self_noise/airfoil_self_noise.dat')
batch_x1=np.loadtxt(data_folder + 'x1.txt')
batch_x2=np.loadtxt(data_folder + 'x2.txt')
batch_x3=np.loadtxt(data_folder + 'x3.txt')
batch_x4=np.loadtxt(data_folder + 'x4.txt')
batch_x5=np.loadtxt(data_folder + 'x5.txt')
batch_y1=np.loadtxt(data_folder + 'y1.txt')


label=batch_y1#+1e-50-1e-50
batch_x=np.column_stack((batch_x1, batch_x2, batch_x3, batch_x4, batch_x5))
batch_y=np.array(batch_y1).reshape(-1, 1)

batch_x_train=batch_x[0:batch_size, :]
batch_y_train=batch_y[0:batch_size, :]
batch_x_test=batch_x[batch_size:, :]
batch_y_test=batch_y[batch_size:, :]

label_train=label[0:batch_size]
label_test=label[batch_size:]

train_hist = []
test_hist = []

with tf.Session() as sess:

    pred = (neural_network) # Apply softmax to logits
    sess.run(init)

    #Training epoch
    for epoch in range(number_epochs):
        sess.run(optimizer, feed_dict={X: batch_x_train, Y: batch_y_train})
        #Display the epoch
        if epoch % 100 == 0:
            print("Epoch:", '%d' % (epoch), end='; ')
            accuracy=tf.losses.mean_squared_error(Y,pred, reduction=tf.losses.Reduction.SUM_OVER_BATCH_SIZE)
            train_loss = accuracy.eval({X: batch_x_train, Y:batch_y_train})
            test_loss = accuracy.eval({X: batch_x_test, Y:batch_y_test})
            train_hist.append(train_loss)
            test_hist.append(test_loss)
            print("Training Loss: ", train_loss, end='; ')
            print("Test Loss: ", test_loss)


    plt.plot(range(0, number_epochs, 100), train_hist, 'b', label='Training')
    plt.plot(range(0, number_epochs, 100), test_hist, 'y', label='Test')
    plt.title('Comparison of Training MSE and Test MSE')
    axes = plt.gca()
    axes.set_ylim([0,100])
    plt.legend(loc="upper right")
    plt.show()

    # Test model
    # print("Prediction:", pred.eval({X: batch_x_train}))
    # training_output=pred.eval({X: batch_x_train})
    # plt.plot(batch_y_train[0:15], 'ro', label='label')
    # plt.plot(training_output[0:15], 'x', label='prediction')
    # plt.ylabel('Labels')
    # plt.legend(loc="upper right")
    # plt.title('Comparison of the prediction of the first 15 training set examples')
    # plt.show()

    # test_output=pred.eval({X: batch_x_test})
    # plt.plot(batch_y_test[0:15], 'ro', label='label')
    # plt.plot(test_output[0:15], 'x', label='prediction')
    # plt.ylabel('Labels')
    # plt.legend(loc="upper right")
    # plt.title('Comparison of the prediction of the first 15 test set examples')
    # plt.show()

    accuracy1 = tf.losses.mean_squared_error(Y,pred, reduction=tf.losses.Reduction.SUM_OVER_BATCH_SIZE)
    print("Final mean squared error: ", accuracy1.eval({X: batch_x_train, Y:batch_y_train}))
    print("Final mean squared error: ", accuracy1.eval({X: batch_x_test, Y:batch_y_test}))

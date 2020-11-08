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
n_hidden2 = 5
n_input = 5
n_output = 1

#Learning parameters
learning_constant = 0.025
number_epochs = 4000
batch_size = 1500

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
loss_op = tf.reduce_mean(tf.math.squared_difference(neural_network,Y))
#loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=neural_network,labels=Y))
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

batch_x=batch_x[0:1500, :]
batch_y=batch_y[0:1500, :]

num_folds = 10
fold_range = int(len(batch_x)/num_folds)

batch_x_fold = {}
batch_y_fold = {}
batch_x_train = []
batch_y_train = []
batch_x_test = []
batch_y_test = []
sum_test_accuracy = 0 #used to calculate average test accuracy

for fold in range(num_folds):
    batch_x_fold[fold] = batch_x[fold*fold_range:(fold+1)*fold_range,:]
    batch_y_fold[fold] = batch_y[fold*fold_range:(fold+1)*fold_range,:]
    #print(batch_x_fold.keys())


def x_distribute(d, keys):
    for i,k in d.items():
        if i != keys:
            batch_x_train.extend(d[i])
        else:
            batch_x_test.extend(d[i])
    return batch_x_train, batch_x_test

def y_distribute(d, keys):
    for i,k in d.items():
        if i != keys:
            batch_y_train.extend(d[i])
        else:
            batch_y_test.extend(d[i])
    return batch_y_train, batch_y_test

with tf.Session() as sess:
    for i, k in enumerate(batch_x_fold.items()):
        
        pred = (neural_network) # Apply softmax to logits
        sess.run(init)

        batch_x_train, batch_x_test = x_distribute(batch_x_fold,i)
        batch_y_train, batch_y_test = y_distribute(batch_y_fold,i)
        print("Test fold = ", i)
        batch_y_train = np.array(batch_y_train).reshape(-1,1)
        batch_y_test = np.array(batch_y_test).reshape(-1,1)

        # print(len(batch_y_test))
        # print(len(batch_y_train))

        #Training epoch
        for epoch in range(number_epochs):
            sess.run(optimizer, feed_dict={X: batch_x_train, Y: batch_y_train})
        
        # Test model
        # print("Prediction:", pred.eval({X: batch_x_train}))
        training_output=pred.eval({X: batch_x_train})

        plt.plot(batch_y_train[0:15], 'ro', training_output[0:15], 'x')
        plt.ylabel('Labels')
        plt.title('Comparison of the prediction of the first 15 training set examples')
        #plt.show()

        test_output=pred.eval({X: batch_x_test})
        accuracy1=tf.losses.mean_squared_error(batch_y_test,test_output, reduction=tf.losses.Reduction.SUM_OVER_BATCH_SIZE)
        test_accuracy = math.sqrt(accuracy1.eval())
        sum_test_accuracy = sum_test_accuracy + test_accuracy
        print("root test accuracy: ", test_accuracy)

        plt.plot(batch_y_test[0:100], 'ro', test_output[0:100], 'x')
        plt.ylabel('Labels')
        plt.title('Comparison of the prediction of the first 15 test set examples')
        plt.show()

        batch_x_train = []
        batch_x_test = []
        batch_y_train = []
        batch_y_test = []


    # print average test accuracy
    print("Averaged root test accuracy for ", num_folds, " folds: ", sum_test_accuracy/num_folds)


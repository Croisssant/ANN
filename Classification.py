import numpy as np
import tensorflow as tf
import math
import logging
logging.basicConfig(level=logging.DEBUG)
import matplotlib.pyplot as plt
import pandas as pd
import os


def data_loader(dataset_path):
    df = pd.read_csv(dataset_path)
    # df = df.sample(frac=1).reset_index(drop=True)
    X = df.iloc[:, 0:-1]
    y = df.iloc[:, -1]
    data_folder = './batch_data/classification/'

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
        X.iloc[:, i].to_csv('{}/x{}.txt'.format(data_folder, i+1), header=False, index=False)

    y.to_csv('{}/y1.txt'.format(data_folder), header=False, index=False)

    return data_folder

#Network parameters
n_hidden1 = 10
n_hidden2 = 10
n_input = 13
n_output = 1 # Still finding a way to change output node to 3

#Learning parameters
learning_constant = 0.2
number_epochs = 1000
batch_size = 1000

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
    out_layer = tf.add(tf.matmul(layer_2, w3),b3)

    return out_layer

#Create model
neural_network = multilayer_perceptron(X)

#Define loss and optimizer
loss_op = tf.reduce_mean(tf.math.squared_difference(neural_network,Y))
#loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=neural_network,labels=Y))
optimizer = tf.train.GradientDescentOptimizer(learning_constant).minimize(loss_op)

#Initializing the variables
init = tf.global_variables_initializer()

data_folder = data_loader('./Data/Wine/refined_data.csv')
batch_x1=np.loadtxt(data_folder + 'x1.txt')
batch_x2=np.loadtxt(data_folder + 'x2.txt')
batch_x3=np.loadtxt(data_folder + 'x3.txt')
batch_x4=np.loadtxt(data_folder + 'x4.txt')
batch_x5=np.loadtxt(data_folder + 'x5.txt')
batch_x6=np.loadtxt(data_folder + 'x6.txt')
batch_x7=np.loadtxt(data_folder + 'x7.txt')
batch_x8=np.loadtxt(data_folder + 'x8.txt')
batch_x9=np.loadtxt(data_folder + 'x9.txt')
batch_x10=np.loadtxt(data_folder + 'x10.txt')
batch_x11=np.loadtxt(data_folder + 'x11.txt')
batch_x12=np.loadtxt(data_folder + 'x12.txt')
batch_x13=np.loadtxt(data_folder + 'x13.txt')
batch_y1=np.loadtxt(data_folder + 'y1.txt')

label=batch_y1#+1e-50-1e-50
batch_x=np.column_stack((batch_x1, batch_x2, batch_x3, batch_x4, batch_x5, batch_x6, batch_x7, batch_x8, batch_x9, batch_x10, batch_x11, batch_x12, batch_x13))
batch_y=np.array(batch_y1).reshape(-1, 1)

batch_x_train=batch_x[0:125, :]
batch_y_train=batch_y[0:125, :]
batch_x_test=batch_x[125:, :]
batch_y_test=batch_y[125:, :]


with tf.Session() as sess:
    sess.run(init)
    #Training epoch
    for epoch in range(number_epochs):

        sess.run(optimizer, feed_dict={X: batch_x_train, Y: batch_y_train})
        #Display the epoch
        if epoch % 100 == 0:
            print("Epoch:", '%d' % (epoch))

    # Test model
    pred = (neural_network) # Apply softmax to logits
    accuracy=tf.keras.losses.MSE(pred,Y)
    print("Accuracy:", accuracy.eval({X: batch_x_train, Y: batch_y_train}))
    #tf.keras.evaluate(pred,batch_x)

    print("Prediction:", pred.eval({X: batch_x_train}))
    output=neural_network.eval({X: batch_x_train})
    plt.plot(batch_y_train[0:10], 'ro', output[0:10], 'x')
    plt.ylabel('some numbers')
    plt.show()

    estimated_class=tf.argmax(pred, 1)#+1e-50-1e-50
    correct_prediction1 = tf.equal(tf.argmax(pred, 1),batch_y_test)
    accuracy1 = tf.reduce_mean(tf.cast(correct_prediction1, tf.float32))
    
    print(accuracy1.eval({X: batch_x}))
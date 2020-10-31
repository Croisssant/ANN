import numpy as np
import tensorflow as tf
import math
import logging
logging.basicConfig(level=logging.DEBUG)
import matplotlib.pyplot as plt
import pandas as pd

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

# ================= Before Normalization (Works) =======================
# Loading data 
df = pd.read_csv('./Data/Wine/refined_data.csv')
df = df.sample(frac=1) # Randomly shuffles each row of data

batch_y = np.array(df.iloc[:, -1]).reshape(-1,1)

batch_x_train = df.iloc[0:100, 0:-1]
batch_y_train = batch_y[0:100]

batch_x_test = df.iloc[101:, 0:-1]
batch_y_test = batch_y[101:]

# ================== After Normalization (Doesn't work yet) =======================
# TypeError: 'DataFrame' objects are mutable, thus they cannot be hashed

df = pd.read_csv('./Data/Wine/refined_data.csv')
df = df.sample(frac=1) # Randomly shuffles each row of data
X = df.iloc[:, 0:-1]

m = X.shape[0] # number of instances
n = X.shape[1] # number of attributes

# Normalization steps
mu = np.mean(X, axis=0)
sigma = np.std(X, axis=0)
batch_x = (X-mu)/sigma
batch_y = np.array(df.iloc[:, -1]).reshape(-1,1) # I think need to change reshape to (-1, 3) if you want to add the node to 3

batch_x_train = batch_x.iloc[:100,:]
batch_y_train = batch_y[0:100]

batch_x_test = batch_x.iloc[101:,:]
batch_y_test = batch_y[101:]


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
    plt.plot(batch_y_train[0:10], 'ro', output[0:10], 'bo')
    plt.ylabel('some numbers')
    plt.show()

    estimated_class=tf.argmax(pred, 1)#+1e-50-1e-50
    correct_prediction1 = tf.equal(tf.argmax(pred, 1),batch_y_test)
    accuracy1 = tf.reduce_mean(tf.cast(correct_prediction1, tf.float32))
    
    print(accuracy1.eval({X: batch_x}))
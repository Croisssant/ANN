import numpy as np
import tensorflow as tf
import math
import logging
#logging.basicConfig(level=logging.DEBUG)
import matplotlib.pyplot as plt
import pandas as pd
import os
import datetime

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
n_hidden1 = 4
n_input = 5
n_output = 1
batch_size = 1150

#Defining the input and the output
X = tf.placeholder("float", [None, n_input])
Y = tf.placeholder("float", [None, n_output])

seed = np.random.randint(1000)

init_random = {
    'b1': tf.get_variable(name='b1', initializer=tf.random_normal([n_hidden1], seed = seed)),
    'b2': tf.get_variable(name='b2', initializer=tf.random_normal([n_output], seed = seed)),
    'w1': tf.get_variable(name='w1', initializer=tf.random_normal([n_input, n_hidden1], seed = seed)),
    'w2': tf.get_variable(name='w2', initializer=tf.random_normal([n_hidden1, n_output], seed = seed)),
}


#DEFINING WEIGHTS AND BIASES
#Biases first hidden layer
b1 = tf.Variable(init_random['b1'])
#Biases second hidden layer
b2 = tf.Variable(init_random['b2'])
#Weights connecting input layer with first hidden layer
w1 = tf.Variable(init_random['w1'])
#Weights connecting first hidden layer with second hidden layer
w2 = tf.Variable(init_random['w2'])


def multilayer_perceptron(input_d):
    #Task of neurons of first hidden layer
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(input_d, w1), b1))
    #Task of neurons of output layer
    out_layer = tf.add(tf.matmul(layer_1, w2), b2) 

    return out_layer

#Create model
neural_network = multilayer_perceptron(X)

#Define loss and optimizer
#loss_op = tf.reduce_mean(tf.math.squared_difference(Y,neural_network))
loss_op = tf.reduce_mean(tf.math.squared_difference(neural_network,Y)) + (0.001 * (tf.reduce_sum(np.absolute(w1)) + tf.reduce_sum(np.absolute(w2))))
#loss_op = tf.reduce_mean(abs(Y - neural_network))
#loss_op = tf.reduce_mean(100 * abs(Y - neural_network) / Y)
#loss_op = tf.reduce_mean(square(log(y_true + 1.) - log(y_pred + 1.)))
#oss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=neural_network,labels=Y))

#Initializing the variables
init = tf.global_variables_initializer()

data_folder = data_loader('./Data/Airfoil_self_noise/airfoil_self_noise.dat')
batch_x1=np.loadtxt(data_folder + 'x1.txt')
batch_x2=np.loadtxt(data_folder + 'x2.txt')
batch_x3=np.loadtxt(data_folder + 'x3.txt')
batch_x4=np.loadtxt(data_folder + 'x4.txt')
batch_x5=np.loadtxt(data_folder + 'x5.txt')
batch_y1=np.loadtxt(data_folder + 'y1.txt')

batch_x=np.column_stack((batch_x1, batch_x2, batch_x3, batch_x4, batch_x5))
batch_y=np.array(batch_y1).reshape(-1, 1)

saver = tf.train.Saver()

train_hist = []
test_hist = []
num_folds = 10
num_eval = 0 #keep track of the number of evaluation
fold_range = int(len(batch_x)/num_folds)

batch_x = batch_x[0:fold_range*num_folds,:] #limit the length of batch_x 
batch_y = batch_y[0:fold_range*num_folds,:] #limit the length of batch_y


print('========== Please select an option ==========')
print('0: No Validation')
print('1: 10-Fold Cross Validation')

ans = int(input('>> '))

if (ans == 1):

    resultsFilename = 'reg-results' + datetime.datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p") + '.csv'
    resultsFile = open(resultsFilename, 'x')
    resultsFile.write('sep=,\n')
    resultsFile.close()
    number_epochs = [500,600,700,800,900]
    learning_rate = [0.05, 0.1, 0.2, 0.3, 0.4]
    with tf.Session() as sess:

        epoch_train_mse = []
        epoch_valid_mse = []

        for ne in number_epochs:
            
            lr_train_mse = []
            lr_valid_mse = []
            # loop for all hyper parameter 
            for param in learning_rate:
                optimizer = tf.train.GradientDescentOptimizer(param).minimize(loss_op)
                print("")
                print("Learning rate: ", param, "of epoch: ", ne)
                print("")

                fold_train_mse = []
                fold_valid_mse = []

                # seperate 10-fold into 9-train and 1-valid
                for fold in range(num_folds):
    
                    batch_x_alpha = batch_x
                    batch_y_alpha = batch_y

                    valid_fold_x = batch_x[fold*fold_range:(fold+1)*fold_range, :]
                    valid_fold_y = batch_y[fold*fold_range:(fold+1)*fold_range, :]        
                    
                    train_fold_x = np.delete(batch_x_alpha, slice(fold*fold_range,(fold+1)*fold_range), axis=0)
                    train_fold_y = np.delete(batch_y_alpha, slice(fold*fold_range,(fold+1)*fold_range), axis=0)
                    
                    sess.run(init)
                    print("-----fold ", fold, " -----")

                    # Training epoch
                    for epoch in range(ne):
                        sess.run(optimizer, feed_dict={X: train_fold_x, Y: train_fold_y})
                        pred = (neural_network) # Apply softmax to logits
     
                    accuracy1 = tf.sqrt(tf.losses.mean_squared_error(Y,pred, reduction=tf.losses.Reduction.SUM_OVER_BATCH_SIZE))
                    print("Mean train squared error for fold: ", accuracy1.eval({X: train_fold_x, Y: train_fold_y}))
                    print("Mean test squared error for fold: ", accuracy1.eval({X: valid_fold_x, Y: valid_fold_y}))
                    fold_train_mse.append(accuracy1.eval({X: train_fold_x, Y: train_fold_y}))
                    fold_valid_mse.append(accuracy1.eval({X: valid_fold_x, Y: valid_fold_y}))

                #----------------------------------------- end of one fold loop --------------------------------------------------#

                lr_train_mse.append(fold_train_mse)
                lr_valid_mse.append(fold_valid_mse)
            #--------------------------------------------- end of one learning rate loop -------------------------------------------------#

            print("=============================================RESULT===============================")
            print("EPOCH: ", ne)
            print(lr_train_mse)

            resultsFile = open(resultsFilename, 'a+')
            resultsFile.write('epoch, ' + str(ne) + '\n')
            resultsFile.write('training data\n')

            i = 0
            for result in lr_train_mse:
                resultsFile.write(str(i) + ', ')
                for resultValue in result:
                    resultsFile.write(str(resultValue) + ', ')
                resultsFile.write('\n')
                i += 1

            resultsFile.write('validation data\n')

            i = 0
            for result in lr_valid_mse:
                resultsFile.write(str(i) + ', ')
                for resultValue in result:
                    resultsFile.write(str(resultValue) + ', ')
                resultsFile.write('\n')
                i += 1

            resultsFile.close()
        #------------------------------------------------------ end of one epoch loop ----------------------------------------------------#
    

elif(ans == 0):
    learning_rate = 0.05
    number_epochs = 900
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss_op)
    with tf.Session() as sess:
        sess.run(init)
        batch_x_train = batch_x[0:1000,:]
        batch_x_test = batch_x[1000:,:]
        batch_y_train = batch_y[0:1000,:]
        batch_y_test = batch_y[1000:,:]

        #Training epoch
        for epoch in range(number_epochs):
            sess.run(optimizer, feed_dict={X: batch_x_train, Y: batch_y_train})
            pred = (neural_network) # Apply softmax to logits
            #Display the epoch
            if epoch % 100 == 0:
                print("Epoch:", '%d' % (epoch), end='; ')
                accuracy= tf.sqrt(tf.losses.mean_squared_error(Y,pred, reduction=tf.losses.Reduction.SUM_OVER_BATCH_SIZE))
                train_loss = accuracy.eval({X: batch_x_train, Y:batch_y_train})
                test_loss = accuracy.eval({X: batch_x_test, Y:batch_y_test})
                train_hist.append(train_loss)
                test_hist.append(test_loss)
                print("Training Loss: ", train_loss, end='; ')
                print("Test Loss: ", test_loss)

        # plt.plot(range(0, number_epochs, 100), train_hist, 'b', label='Training')
        # plt.plot(range(0, number_epochs, 100), test_hist, 'y', label='Test')
        # plt.title('Comparison of Training MSE and Test MSE')
        # axes = plt.gca()
        # axes.set_ylim([0,100])
        # plt.legend(loc="upper right")
        # plt.show()

        test_output=pred.eval({X: batch_x_test})
        plt.plot(batch_y_test[0:15], 'ro', label='label')
        plt.plot(test_output[0:15], 'x', label='prediction')
        plt.ylabel('Labels')
        plt.legend(loc="upper right")
        plt.title('Comparison of the prediction of the first 15 test set examples')
        plt.show()

        accuracy1 = tf.sqrt(tf.losses.mean_squared_error(Y,pred, reduction=tf.losses.Reduction.SUM_OVER_BATCH_SIZE))
        print("Final root mean squared error: ", accuracy1.eval({X: batch_x_train, Y:batch_y_train}))
        print("Final root mean squared error: ", accuracy1.eval({X: batch_x_test, Y:batch_y_test}))
        
        saver.save(sess, 'my_test_model')
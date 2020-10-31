import numpy as np
import tensorflow as tf
import math
import logging
# logging.basicConfig(level=logging.DEBUG)
import matplotlib.pyplot as plt
import pandas as pd


df = pd.read_csv('./Data/Wine/refined_data.csv')
df = df.sample(frac=1)
X = df.iloc[:, 0:-1]
print('This is X \n', X)

# print('This is type X', X.keys())

m = X.shape[0] # number of instances
n = X.shape[1] # number of attributes

mu = pd.DataFrame.mean(X, axis=0)
sigma = pd.DataFrame.std(X, axis=0)
batch_x = (X-mu)/sigma

# print('This is type batch_x\n', batch_x.keys())

print(batch_x)
print('ANS=================')
print('This is batch_x: \n', batch_x.iloc[:,:])

# print(batch_x['Alcohol'])
# print(batch_x[:100])
batch_x_train = batch_x[:100]

batch_x_test = batch_x[101:]

# x1 = df.iloc[:, 0]

# batch_x_train = df.iloc[0:100, 0:-1]
# batch_y_train = df.iloc[0:100, -1]

# batch_x_test = df.iloc[101:, 0:-1]
# batch_y_test = df.iloc[101:, -1]

# print('=======================')
# print('x_train')
# print(batch_x_train)

# print('=======================')
# print('batch_y_train')
# print(batch_y_train)

# print('=======================')
# print('batch_x_test')
# print(batch_x_test)

# print('=======================')
# print('batch_y_test')
# print(batch_y_test)


# dataset_path = './Data/Wine/refined_data.csv'
# df = pd.read_csv(dataset_path)
# x1 = df.iloc[:, 0]
# x2 = df.iloc[:, 1]
# x3 = df.iloc[:, 2]
# y = df.iloc[:, -1]

# k = df.iloc[0:5, 0:-1]
# print('==================')
# print(k)
# print('==================')
# print('x1:')
# print(x1)
# print('x2:')
# print(x2)
# print('x3:')
# print(x3)
# print('y:')
# print(y)

# df = df.sample(frac=1)
# x1 = df.iloc[:, 0]
# x2 = df.iloc[:, 1]
# x3 = df.iloc[:, 2]
# y = df.iloc[:, -1]
# print('x1:')
# print(x1)
# print('x2:')
# print(x2)
# print('x3:')
# print(x3)
# print('y:')
# print(y)

# # Writing data into attribute files
# def data_loader(dataset_path):
    
#     data_folder = './batch_data/regression/'

#     m = X.shape[0] # number of instances
#     n = X.shape[1] # number of attributes

#     mu = np.mean(X, axis=0)
#     sigma = np.std(X, axis=0)
#     X = (X-mu)/sigma

#     if not os.path.exists(data_folder): 
#         os.makedirs(data_folder)

#     for i in range(n):
#         X[i].to_csv('{}/x{}.txt'.format(data_folder, i+1), header=False, index=False)

#     y.to_csv('{}/y1.txt'.format(data_folder), header=False, index=False)

#     return data_folder


# data_folder = data_loader('./Data/Airfoil_self_noise/airfoil_self_noise.dat')
# batch_x1=np.loadtxt(data_folder + 'x1.txt')
# batch_x2=np.loadtxt(data_folder + 'x2.txt')
# batch_x3=np.loadtxt(data_folder + 'x3.txt')
# batch_x4=np.loadtxt(data_folder + 'x4.txt')
# batch_x5=np.loadtxt(data_folder + 'x5.txt')
# batch_y1=np.loadtxt(data_folder + 'y1.txt')
import struct as st
import os
import numpy as np
import random
from matplotlib import pyplot as plt
from numpy import linalg as LA
import idx2numpy

def add_noise(Data, level):

    for i in range(len(Data)):
        noise = np.random.normal(10, 10, 28*28)
        Data[i] = np.add(Data[i], float(level)*noise)
    return Data


def pca(data):
    cov_mat = np.cov(data, rowvar=False)
    # print(cov_mat.shape)
    # for i in range(len(Data[:,0])):
    #     Data[i,:]=Data[i,:]-mean_data
    # print(mean_data)

    Eigen_vals, Eigen_vecs = LA.eig(cov_mat)
    Eigen_vals=np.real(Eigen_vals)
    Eigen_vecs=np.real(Eigen_vecs)
    print(Eigen_vecs.shape, Eigen_vals.shape)

    temp = []
    for i in range(len(Eigen_vals)):
        temp.append([Eigen_vals[i],Eigen_vecs[:,i]])

    temp.sort(key = lambda x:x[0], reverse=True)

    # Original Image
    img = np.reshape(mean_data,(28,28))
    plt.imshow(img,'gray')
    plt.show()

    Eigen_clip = []
    for i in range(10):
        Eigen_clip.append(temp[i][1])


    Eigen_clip = np.asarray(Eigen_clip)
    img = np.reshape(Eigen_clip[0,:],(28,28))
    # plt.subplot(np.reshape(Eigen_clip[0,:],(28,28)),'gray')
    # plt.imshow(img,'gray')
    # plt.show()
    Eigen_clip = np.transpose(Eigen_clip)
    y = np.dot(Data,Eigen_clip)
    Eigen_clip_t = np.transpose(Eigen_clip)
    print(y.shape)
    print(Eigen_clip_t.shape)
    y = np.dot(y, Eigen_clip_t)

    img = np.reshape(y[0,:],(28,28))
    print(img.shape)
    plt.imshow(np.clip(img,0,255),'gray')
    plt.show()

if __name__ == '__main__':
    data = idx2numpy.convert_from_file('train-images-idx3-ubyte')
    label = idx2numpy.convert_from_file('train-labels-idx1-ubyte')

    digit=[[] for i in range(10)]

    for i in range(len(data)):
        digit[label[i]].append(data[i].reshape(28*28))
    digit = np.asarray(digit)

    # Data = add_noise(Data,0.2)

    inp = int(input('Please enter the digit: '))

    Data = digit[inp]
    Data = np.asarray(Data)
    # print(Data.shape)
    mean_data = np.mean(Data, axis=0, dtype=float)
    # print(mean_data.shape)
    pca(Data)

    Data = add_noise(Data, 0.2)

    pca(Data)


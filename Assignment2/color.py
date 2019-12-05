import numpy as np
import matplotlib.pyplot as plt
import math
import random
from PIL import Image

K = 10

img = Image.open('image.jpg')
data = np.array(img)
out_img = data
mean = np.random.randint(low=0, high=255, size=(K,3))
print(mean)

def euclidean_distance(sample, mean):
    d = pow(sample[0]-mean[0], 2) + pow(sample[1]-mean[1], 2) + pow(sample[2]-mean[2], 2)
    return d

precost = -1
cost = 0
ptr = 1

while abs(cost-precost) > 0.001:
    precost = cost
    cluster_no = np.zeros(shape=(data.shape[0], data.shape[1]), dtype=int)

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            minimum = 100000000
            min_index = -1
            dist_from_cluster_mean = np.zeros(shape=K)
            for k in range(K):
                dist_from_cluster_mean[k] = euclidean_distance(data[i][j], mean[k])
                if(dist_from_cluster_mean[k] < minimum):
                    minimum = dist_from_cluster_mean[k]
                    min_index = k
            cluster_no[i][j] = min_index
    print(cluster_no.shape)
    

    dataset_clusterwise = np.zeros(K, dtype=int)
    sum = np.zeros(shape=(K,3))

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            dataset_clusterwise[cluster_no[i][j]] = dataset_clusterwise[cluster_no[i][j]] + 1
            sum[cluster_no[i][j]] = sum[cluster_no[i][j]] + data[i][j]


    for i in range(K):
        mean[i] = sum[i]/dataset_clusterwise[i]
    print(mean)

    cost = 0
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            cost += pow(data[i][j][0]-mean[cluster_no[i][j]][0], 2) + pow(data[i][j][1]-mean[cluster_no[i][j]][1], 2) + pow(data[i][j][2]-mean[cluster_no[i][j]][2], 2)

    print("cost")
    print(cost)

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            print(out_img[i][j].shape, mean.shape, cluster_no[i][j])
            out_img[i][j] = mean[cluster_no[i][j]]

    img_name = 'img'+ str(ptr) +'.jpg'
    ptr += 1
    im = Image.fromarray(out_img)
    im.save(img_name)


import data as D
import utils as U
import matplotlib.pyplot as plt
import numpy as np
import os
import math
import warnings
warnings.simplefilter("ignore")

def check(x, y, h):
    # dist = (x[0]-y[0])**2 + (x[1]-y[1])**2
    # if dist <= h*h:
    #     return True
    # return False
    if x[0] >= y[0] - h and x[0] <= y[0] + h and x[1] >= y[1] - h and x[1] <= y[1] + h:
        return True
    return False


def hypercube(data, datapoints, total):
    cnt = 0.0
    h = 0.3
    for point in datapoints:
        if check(data, point, h):
            cnt += 1
    return (cnt) / ( total * 4*h*h) 
    # return cnt / ( total * math.pi * h * h)


def decision_boundary(plt, x1, x2, x3, total):
    range_x = [-3, 3]
    range_y = [-3, 3]
    temp1 = [[], []]
    temp2 = [[], []]
    temp3 = [[], []]
    val = 0.1
    start = float(range_x[0])
    while start < float(range_x[1]):
        start_j = float(range_y[0])
        while start_j < float(range_y[1]):
            p1 = hypercube([start, start_j], x1, total)
            p2 = hypercube([start, start_j], x2, total)
            p3 = hypercube([start, start_j], x3, total)
            if p1 > p2 and p1 > p3:
                temp1[0].append(start)
                temp1[1].append(start_j)
            elif p2 > p1 and p2 > p3:
                temp2[0].append(start)
                temp2[1].append(start_j)
            else:
                temp3[0].append(start)
                temp3[1].append(start_j)

            start_j += val
        start += val

    plt.scatter(temp1[0], temp1[1], color="pink", zorder=-1)
    plt.scatter(temp2[0], temp2[1], color="yellow", zorder=-1)
    plt.scatter(temp3[0], temp3[1], color="orange", zorder=-1)


def parzen(x1, x2, x3, plt):

    all_ = x1 + x2 + x3
    temp1 = [[], []]
    temp2 = [[], []]
    temp3 = [[], []]

    for point in all_:
        p1 = hypercube(point, x1, len(all_))
        p2 = hypercube(point, x2, len(all_))
        p3 = hypercube(point, x3, len(all_))

        if p1 > p2 and p1 > p3:
            temp1[0].append(point[0])
            temp1[1].append(point[1])
        elif p2 > p1 and p2 > p3:
            temp2[0].append(point[0])
            temp2[1].append(point[1])
        else:
            temp3[0].append(point[0])
            temp3[1].append(point[1])
    
    decision_boundary(plt, x1, x2, x3, len(all_))
    plt.scatter(temp1[0], temp1[1], color="blue", zorder=-1)
    plt.scatter(temp2[0], temp2[1], color="red", zorder=-1)
    plt.scatter(temp3[0], temp3[1], color="green", zorder=-1)


def build_conf(conf, a, b, c, x1, x2, x3):
    total = len(x1+x2+x3)
    for point in a:
        p1 = hypercube(point, x1, total)
        p2 = hypercube(point, x2, total)
        p3 = hypercube(point, x3, total)

        if p1 > p2 and p1 > p3:
            conf[0][0] += 1
        elif p2 > p1 and p2 > p3:
            conf[0][1] += 1
        else:
            conf[0][2] += 1

    for point in b:
        p1 = hypercube(point, x1, total)
        p2 = hypercube(point, x2, total)
        p3 = hypercube(point, x3, total)

        if p1 > p2 and p1 > p3:
            conf[1][0] += 1
        elif p2 > p1 and p2 > p3:
            conf[1][1] += 1
        else:
            conf[1][2] += 1
    
    for point in c:
        p1 = hypercube(point, x1, total)
        p2 = hypercube(point, x2, total)
        p3 = hypercube(point, x3, total)

        if p1 > p2 and p1 > p3:
            conf[2][0] += 1
        elif p2 > p1 and p2 > p3:
            conf[2][1] += 1
        else:
            conf[2][2] += 1



if __name__ == '__main__':
    
    x1 = D.get_data('Data/Class1.txt')
    x2 = D.get_data('Data/Class2.txt')
    x3 = D.get_data('Data/Class3.txt')

    x1_train, x1_test = U.split(x1, 0.1)
    x2_train, x2_test = U.split(x2, 0.1)
    x3_train, x3_test = U.split(x3, 0.1)

    plt.figure()
    parzen(x1_train, x2_train, x3_train, plt)
    plt.savefig('parzen.png')

    conf_mat = [[0,0,0],[0,0,0],[0,0,0]]

    build_conf(conf_mat, x1_test, x2_test, x3_test, x1_train, x2_train, x3_train)

    print(conf_mat)
    print("Accuracy: ", U.accuracy(conf_mat))
    print("Precision for class 1: ", U.precision(conf_mat, 0))
    print("Precision for class 2: ", U.precision(conf_mat, 1))
    print("Precision for class 3: ", U.precision(conf_mat, 2))
    print("Recall for class 1: ", U.precision(conf_mat, 0))
    print("Recall for class 2: ", U.precision(conf_mat, 1))
    print("Recall for class 3: ", U.precision(conf_mat, 2))
    print("F-Score for class 1: ", U.precision(conf_mat, 0))
    print("F-Score for class 2: ", U.precision(conf_mat, 1))
    print("F-Score for class 3: ", U.precision(conf_mat, 2))








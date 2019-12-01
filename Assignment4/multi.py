import data as D
import utils as U
import matplotlib.pyplot as plt
import numpy as np
import os
import warnings
warnings.simplefilter("ignore")

def dot(w, x):
    return w[0]*x[0] + w[1]*x[1] + w[2]*x[2]

def add(w, x, y):
    for i in range(len(w)):
        w[i] = w[i] + y*x[i]

def decision_boundary(plt, w1, w2, w3):
    range_x = [-15,25]
    range_y = [-20,20]
    points_x = [[],[],[]]
    points_y = [[],[],[]]

    val = 0.1
    start = float(range_x[0])
    while start < float(range_x[1]):
        start_j = float(range_y[0])
        while start_j < float(range_y[1]):
            val1 = dot(w1, [1, start, start_j])
            val2 = dot(w2, [1, start, start_j])
            val3 = dot(w3, [1, start, start_j])

            if val1 > val2 and val1 > val3:
                points_x[0].append(start)
                points_y[0].append(start_j)
        
            elif val2 > val1 and val2 > val3:
                points_x[1].append(start)
                points_y[1].append(start_j)
            else:
                points_x[2].append(start)
                points_y[2].append(start_j)
                
            start_j += val
        start += val

    plt.scatter(points_x[0],points_y[0],color="yellow",zorder=-1)
    plt.scatter(points_x[1],points_y[1],color="red",zorder=-1)
    plt.scatter(points_x[2],points_y[2],color="blue",zorder=-1)

def perceptron(w, x, y, z, alpha, iterations, cur):
    cost = []
    iter_ = []

    for i in range(iterations):
        cost_ = 0.0
        del_w = [0.0, 0.0, 0.0]

        for j in range(len(x)):
            val = dot(x[j], w)
            if val < 0:
                cost_ -= val
                add(del_w, x[j], alpha)

        for j in range(len(y)):
            val = dot(y[j], w)
            if val > 0:
                cost_ += val
                add(del_w, y[j], -alpha)
        
        for j in range(len(z)):
            val = dot(z[j], w)
            if val > 0:
                cost_ += val
                add(del_w, z[j], -alpha)

        add(w, del_w, 1)
        # print(w)
        cost.append(cost_)
        iter_.append(i+1)

    plt.figure()
    plt.plot(iter_, cost, color='green')
    plt.savefig('plots_multi/cost_' + str(cur) + '.png')

def get_conf(x1, x2, x3, w1, w2, w3,plt):
    points_x = [[],[],[]]
    points_y = [[],[],[]]
    conf_mat = [[0,0,0],[0,0,0],[0,0,0]]

    for i in range(len(x1)):
        val1 = dot(w1, x1[i])
        val2 = dot(w2, x1[i])
        val3 = dot(w3, x1[i])

        if val1 > val2 and val1 > val3:
            points_x[0].append(x1[i][1])
            points_y[0].append(x1[i][2])
            conf_mat[0][0] += 1
        
        elif val2 > val1 and val2 > val3:
            points_x[1].append(x1[i][1])
            points_y[1].append(x1[i][2])
            conf_mat[0][1] += 1
        else:
            points_x[2].append(x1[i][1])
            points_y[2].append(x1[i][2])
            conf_mat[0][2] += 1
    
    for i in range(len(x2)):
        val1 = dot(w1, x2[i])
        val2 = dot(w2, x2[i])
        val3 = dot(w3, x2[i])

        if val1 > val2 and val1 > val3:
            points_x[0].append(x2[i][1])
            points_y[0].append(x2[i][2])
            conf_mat[1][0] += 1
        
        elif val2 > val1 and val2 > val3:
            points_x[1].append(x2[i][1])
            points_y[1].append(x2[i][2])
            conf_mat[1][1] += 1
        else:
            points_x[2].append(x2[i][1])
            points_y[2].append(x2[i][2])
            conf_mat[1][2] += 1
    
    for i in range(len(x3)):
        val1 = dot(w1, x3[i])
        val2 = dot(w2, x3[i])
        val3 = dot(w3, x3[i])

        if val1 > val2 and val1 > val3:
            points_x[0].append(x3[i][1])
            points_y[0].append(x3[i][2])
            conf_mat[2][0] += 1
        
        elif val2 > val1 and val2 > val3:
            points_x[1].append(x3[i][1])
            points_y[1].append(x3[i][2])
            conf_mat[2][1] += 1
        else:
            points_x[2].append(x3[i][1])
            points_y[2].append(x3[i][2])
            conf_mat[2][2] += 1
    
    plt.scatter(points_x[0],points_y[0],color="black",zorder=-1)
    plt.scatter(points_x[1],points_y[1],color="white",zorder=-1)
    plt.scatter(points_x[2],points_y[2],color="green",zorder=-1)
    return conf_mat

        
if __name__ == '__main__':
    x1 = D.get_data('Data1/Class1.txt')
    x2 = D.get_data('Data1/Class2.txt')
    x3 = D.get_data('Data1/Class3.txt')

    if not os.path.isdir("plots_multi"):
        os.mkdir("plots_multi")

    w1 = [200.0,-200.0,200.0]
    w2 = [-200.0,200.0,200.0]
    w3 = [200.0,200.0,-200.0]

    perceptron(w1, x1, x2, x3, 1, 100, 1)
    perceptron(w2, x2, x3, x1, 1, 100, 2)
    perceptron(w3, x3, x1, x2, 1, 100, 3)

    plt.figure()
    decision_boundary(plt, w1, w2, w3)
    conf_mat = get_conf(x1,x2,x3,w1,w2,w3,plt)
    plt.savefig('plots_multi/decision_boundary.png')

    print(conf_mat)
    print("Accuracy: ", U.accuracy(conf_mat))
    print("Precision for class 1: ", U.precision(conf_mat, 0))
    print("Precision for class 2: ", U.precision(conf_mat, 1))
    print("Precision for class 3: ", U.precision(conf_mat, 2))
    print("Recall for class 1: ", U.recall(conf_mat, 0))
    print("Recall for class 2: ", U.recall(conf_mat, 1))
    print("Recall for class 3: ", U.recall(conf_mat, 2))
    print("F-Score for class 1: ", U.f_score(conf_mat, 0))
    print("F-Score for class 2: ", U.f_score(conf_mat, 1))
    print("F-Score for class 3: ", U.f_score(conf_mat, 2))



    

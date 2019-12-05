import data as D
import matplotlib.pyplot as plt
import numpy as np
import utils as U
import os
import warnings
warnings.simplefilter("ignore")

def dot(w, x):
    return w[0]*x[0] + w[1]*x[1] + w[2]*x[2]

def add(w, x, y):
    for i in range(len(w)):
        w[i] = w[i] + y*x[i]

def decision_boundary(plt):
    range_x = [-15,15]
    range_y = [-20,20]
    points_x = [[],[]]
    points_y = [[],[]]
    val = 0.1
    start = float(range_x[0])
    while start < float(range_x[1]):
        start_j = float(range_y[0])
        while start_j < float(range_y[1]):
            dot = w[0]*1 + w[1]*start + w[2]*start_j
            if dot < 0:
                points_x[1].append(start)
                points_y[1].append(start_j)
            else:
                points_x[0].append(start)
                points_y[0].append(start_j)
            start_j += val
        start += val

    plt.scatter(points_x[0],points_y[0],color="yellow",zorder=-1)
    plt.scatter(points_x[1],points_y[1],color="red",zorder=-1)


def perceptron(w, x1, x2, alpha, iterations):
    cost = []
    iter_ = []

    for i in range(iterations):
        cost_ = 0.0
        del_w = [0.0, 0.0, 0.0]
        temp1 = [[],[]]
        temp2 = [[],[]]

        for j in range(len(x1)):
            val = dot(x1[j], w)
            if val < 0:
                cost_ -= val
                add(del_w, x1[j], alpha)
                temp2[0].append(x1[j][1])
                temp2[1].append(x1[j][2])
            else:
                temp1[0].append(x1[j][1])
                temp1[1].append(x1[j][2])

        for j in range(len(x2)):
            val = dot(x2[j], w)
            if val > 0:
                cost_ += val
                add(del_w, x2[j], -alpha)
                temp1[0].append(x2[j][1])
                temp1[1].append(x2[j][2])
            else:
                temp2[0].append(x2[j][1])
                temp2[1].append(x2[j][2])

        plt.figure()
        decision_boundary(plt)
        plt.scatter(temp1[0],temp1[1],color="pink",zorder=-1)
        plt.scatter(temp2[0],temp2[1],color="blue",zorder=-1)

        add(w, del_w, 1)
        plt.savefig('plots/iteration' + str(i) + '.png')
        print(w)
        cost.append(cost_)
        iter_.append(i+1)
    plt.figure()
    plt.plot(iter_, cost, color='green')
    plt.savefig('plots/Cost.png')
    


if __name__ == '__main__':
    x1 = D.get_data('Data1/Class3.txt')
    x2 = D.get_data('Data1/Class2.txt')
    x1_train, x1_test = U.split(x1, 0.1)
    x2_train, x2_test = U.split(x1, 0.1)
    if not os.path.isdir("plots"):
        os.mkdir("plots")

    w = [-0.1,0.1,-0.1]
    print(w)

    # perceptron(w, x1, x2, 0.7, 5)
    perceptron(w, x1_train, x2_train, 0.3, 20)

    conf_mat = [[0,0],[0,0]]
    for i in range(len(x1_test)):
        val = dot(w, x1_test[i])
        if val<0:
            conf_mat[0][1] += 1
        else:
            conf_mat[0][0] += 1
    
    for i in range(len(x2_test)):
        val = dot(w, x1_test[i])
        if val>0:
            conf_mat[1][0] += 1
        else:
            conf_mat[1][1] += 1
    print(conf_mat)
    # providing such low alpha as the algo is learning too fast
    # to show gradual formation of decision surface keep alpha low(0.3) and iteration around(20)
    # alpha 0.1 and iterations 40 saved
    

        
import data as D
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


def do(w, x, p, del_w):
    for i in range(len(x[p])):
        for k in range(3):
            val = dot(x[p][j], w[k])
            if k == p:
                if val < 0:
                    add(del_w[k], x[p][j], alpha)
                else:
                    pass
            else:
                if val > 0:
                    add(del_w[k], x[p][j], -alpha)
                else:
                    pass
        

def perceptron(w, x):

    for i in range(iterations):
        del_w = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]

        do(w, x, 0)

if __name__ == '__main__':
    x1 = D.get_data('Data1/Class1.txt')
    x2 = D.get_data('Data1/Class2.txt')
    x3 = D.get_data('Data1/Class3.txt')
    x = [x1, x2, x3]

    if not os.path.isdir("plots_multi"):
        os.mkdir("plots_multi")

    w = [[-200.0,200.0,-200.0],[200.0,-200.0,-200.0], [-200.0,-200.0,200.0]]
    

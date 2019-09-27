import math
import numpy as np 

# Function to Calculate mean
def mean(data):
    mu = [0]*len(data)

    for i in range(len(data)):
        mu[i] = sum(data[i])/float(len(data[i]))

    return mu

# Function to calculate variance
def variance(data):
    mu = mean(data)
    var = [0]*len(data)

    for i in range(len(data)):
        for j in range(len(data[i])):
            var[i] += pow(data[i][j]-mu[i], 2)
        var[i] /= float(len(data[i])-1)
    
    return var

def avg_var(data):
    var = variance(data)
    avg_variane = sum(var)/float(len(var))
    return avg_variane

def euclidean_norm(v, mu):
    val = 0
    for i in range(len(mu)):
        val += (v[i]-mu[i])**2
    
    return math.sqrt(val)


def discriminant_func(v, mu, var):
    return (-1 * pow(euclidean_norm(v, mu), 2) ) / (2*var)


def accuracy(conf_matrix):
    cnt = 0
    neg = 0
    for i in range(3):
        for j in range(3):
            cnt += conf_matrix[i][j]
            if i != j:
                neg += conf_matrix[i][j]

    return ((cnt-neg)/float(cnt))*100

def precision(conf_matrix, i):
    cnt = 0
    pos = 0

    for j in range(3):
        cnt += conf_matrix[i][j]
        if i == j:
            pos = conf_matrix[i][j]

    return pos/cnt

def recall(conf_matrix, j):
    cnt = 0
    pos = 0

    for i in range(3):
        cnt += conf_matrix[i][j]
        if i == j:
            pos = conf_matrix[i][j]

    return pos/cnt

def f_score(conf_matrix, i):
    P = precision(conf_matrix, i)
    R = recall(conf_matrix, i)

    return (2*P*R)/(P+R)

def mat_inverse(arr):
    mat = np.array(arr)
    return np.linalg.inv(mat)

def mat_mult(mat1, mat2):
    a = np.array(mat1)
    b = np.array(mat2)
    return np.dot(a,b)
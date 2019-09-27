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

def get_cov(data, mean, x, y):
    t = 0
    for i in range(len(data[0])):
        t += (data[x][i]-mean[x]) * (data[y][i]-mean[y])
    
    t /= len(data[0])
    return t

def covariance_mat(data):
    mu = mean(data)
    conv = [[0,0],[0,0]]
    # for i in range(len(data[0])):
    #     for j in range(len(data[0])):
    #         cov[i][j] = get_cov(data, mean, i, j)
    conv[0][0] = get_cov(data, mu, 0, 0)
    conv[1][1] = get_cov(data, mu, 1, 1)
    conv[1][0] = get_cov(data, mu, 1, 0)
    conv[0][1] = conv[1][0]
    return conv



def avg_var(data):
    var = variance(data)
    avg_variane = sum(var)/float(len(var))
    return avg_variane


def mat_inverse(arr):
    mat = np.array(arr)
    return np.linalg.inv(mat)

def mat_mul(mat1, mat2):
    a = np.array(mat1)
    b = np.array(mat2)
    return np.dot(a,b)

def euclidean_norm(v, mu):
    val = 0
    for i in range(len(mu)):
        val += (v[i]-mu[i])**2
    
    return math.sqrt(val)


def discriminant_func(v, mu, conv_mat, case):

    if case == 1:
        return (-1 * pow(euclidean_norm(v, mu), 2) ) / (2*conv_mat[0][0])
        
    elif case == 2:
        temp = [v[0]-mu[0], v[1]-mu[0]]
        return (-1 * mat_mul(mat_mul(temp, conv_mat), temp) ) / 2

    inv = mat_inverse(conv_mat)
    mean_np = np.asarray([mean])
    meannpt = np.transpose(meannp)
    x = np.array([v])
    xt = np.transpose(x)
    temp = np.dot(x,inv)
    temp = np.dot(temp, xt)
    val = temp[0][0]
    val1 = -val/2
    temp = np.dot(inv, np.transpose(meannp))
    



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


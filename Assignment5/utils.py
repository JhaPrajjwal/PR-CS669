import random

def split(X, ratio = 0.1):
    random.shuffle(X)
    len_ = int((float)(1.0-ratio)*len(X))
    return X[:len_], X[len_:]

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

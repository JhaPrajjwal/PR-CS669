import random

def split(X, ratio = 0.1):
    random.shuffle(X)
    len_ = int((float)(1.0-ratio)*len(X))
    return X[:len_], X[len_:]

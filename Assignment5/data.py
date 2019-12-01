import random

def get_data(filename):  
    X = []
    with open(filename) as f:
        data = f.read().splitlines()
        for line in data:
            x, y = line.split()
            X.append([float(x), float(y)])

    return X






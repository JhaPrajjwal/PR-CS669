import random

def get_data(filename):  
    X = []
    Y = []
    with open(filename) as f:
        data = f.read().splitlines()
        for line in data:
            x, y = line.split()
            X.append(float(x))
            Y.append(float(y))

    random.shuffle(X)
    random.shuffle(Y)
    train_X = X[:(3*len(X))//4]
    test_X = X[(3*len(X))//4:]
    train_Y = Y[:(3*len(Y))//4]
    test_Y = Y[(3*len(Y))//4:]

    return [train_X, train_Y], [test_X, test_Y]






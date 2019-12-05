import data as D
import utils as U
import matplotlib.pyplot as plt
import numpy as np
import os
import warnings
warnings.simplefilter("ignore")
from scipy.stats import multivariate_normal as mvn
from sklearn.cluster import KMeans

def init(K, X):
    kmeans = KMeans(n_clusters= K, init="k-means++", max_iter=500, algorithm = 'auto')
    fit = kmeans.fit(X)
    pred = kmeans.predict(X)
    
    d = X.shape[1]
    labels = np.unique(pred)
    init_means = np.zeros((K, d))
    init_cov = np.zeros((K, d, d))
    init_pi = np.zeros(K)
        
    cnt = 0
    for label in labels:
        ids = np.where(pred == label) # returns indices
        init_pi[cnt] = len(ids[0]) / X.shape[0]
        init_means[cnt,:] = np.mean(X[ids], axis = 0)
        de_meaned = X[ids] - init_means[cnt,:]
        Nk = X[ids].shape[0]
        init_cov[cnt,:, :] = np.dot(init_pi[cnt] * de_meaned.T, de_meaned) / Nk
        cnt += 1
    
    return init_means, init_cov, init_pi


def e_step(X, mu, pi, sigma, K):
    N = X.shape[0] 
    gamma = np.zeros((N, K))
    const_c = np.zeros(K)
        
    for k in range(K):
        gamma[:,k] = pi[k] * mvn.pdf(X, mu[k,:], sigma[k])

    gamma_norm = np.sum(gamma, axis=1)[:,np.newaxis]
    gamma /= gamma_norm
    return gamma


def m_step(X, gamma, sigma, mu, pi, K):
    N = X.shape[0] 
    d = X.shape[1]
    pi = np.mean(gamma, axis = 0)
    mu = np.dot(gamma.T, X) / np.sum(gamma, axis = 0)[:,np.newaxis]

    for k in range(K):
        x = X - mu[k, :]
        gamma_diag = np.diag(gamma[:,k])
        x_mu = np.matrix(x)
        gamma_diag = np.matrix(gamma_diag)
        sigma_c = x.T * gamma_diag * x
        sigma[k,:,:]=(sigma_c) / np.sum(gamma, axis = 0)[:,np.newaxis][k]

    return pi, mu, sigma


def loss_function(X, pi, mu, sigma, gamma, K):
    N = X.shape[0]
    loss = np.zeros((N, K))

    for k in range(K):
        dist = mvn(mu[k], sigma[k],allow_singular=True)
        loss[:,k] = gamma[:,k] * (np.log(pi[k]+0.00001)+dist.logpdf(X)-np.log(gamma[:,k]+0.000001))
    loss = np.sum(loss)
    # return abs(loss)
    return loss

def gmm(K, X, iterations):
    mu, sigma, pi = init(K, X)
    cost = []
    x_ = []
    for run in range(iterations):  
        gamma  = e_step(X, mu, pi, sigma, K)
        pi, mu, sigma = m_step(X, gamma, sigma, mu, pi, K)
        loss = loss_function(X, pi, mu, sigma, gamma, K)  
        cost.append(loss)
        x_.append(run)
    
    x = []
    y = []
    for i in range(len(X)):
        x.append(X[i][0])
        y.append(X[i][1])

    for i in range(K):
        U.plot_contour(mu[i],sigma[i],x,y,plt)
    

    plt.figure()
    plt.plot(x_, cost, color='green')
    plt.savefig('GMM_cost.png')
    plt.close()

    return mu, sigma, pi

def predict(K, X, mu, sigma, pi):
    labels = np.zeros((X.shape[0], K))
    for k in range(K):
        labels [:,k] = pi[k] * mvn.pdf(X, mu[k,:], sigma[k])
    labels = labels.argmax(1)
    return labels 


if __name__ == '__main__':
    
    x1 = D.get_data('Data/Class1.txt')
    x2 = D.get_data('Data/Class2.txt')
    x3 = D.get_data('Data/Class3.txt')

    X = np.concatenate((x1,x2))
    X = np.concatenate((X,x3))

    K = 3
    
    plt.figure()
    # mu, sigma, pi = gmm(3, X, 100)

    x1 = np.asarray(x1)
    x2 = np.asarray(x2)
    x3 = np.asarray(x3)
    # label = predict(3, X, mu, sigma, pi)
    # label1 = predict(3, x1, mu, sigma, pi)
    # label2 = predict(3, x2, mu, sigma, pi)
    # label3 = predict(3, x3, mu, sigma, pi)

    mu1, sigma1, pi1 = gmm(K, x1, 200)
    mu2, sigma2, pi2 = gmm(K, x2, 200)
    mu3, sigma3, pi3 = gmm(5, x3, 200)

    x1_test = D.get_data('Data/Class1.txt')
    x2_test = D.get_data('Data/Class2.txt')
    x3_test = D.get_data('Data/Class3.txt')

    _, x1_test = U.split(x1_test, 0.05)
    _, x2_test = U.split(x2_test, 0.05)
    _, x3_test = U.split(x3_test, 0.05)


    temp1 = [[], []]
    temp2 = [[], []]
    temp3 = [[], []]
    for i in range(len(X)):
        sum1 = 0.0
        sum2 = 0.0
        sum3 = 0.0

        for j in range(K):
            sum1 += pi1[j]*mvn.pdf(X[i], mu1[j], sigma1[j])
        for j in range(K):
            sum2 += pi2[j]*mvn.pdf(X[i], mu2[j], sigma2[j])
        for j in range(K):
            sum3 += pi3[j]*mvn.pdf(X[i], mu3[j], sigma3[j])
        
        if sum1 >= sum2 and sum1 >= sum3:
            temp1[0].append(X[i][0])
            temp1[1].append(X[i][1])
        elif sum2 >= sum3 and sum2 >= sum1:
            temp2[0].append(X[i][0])
            temp2[1].append(X[i][1])
        else:
            temp3[0].append(X[i][0])
            temp3[1].append(X[i][1])

    
    plt.scatter(temp1[0], temp1[1], color="pink", zorder=-1)
    plt.scatter(temp2[0], temp2[1], color="yellow", zorder=-1)
    plt.scatter(temp3[0], temp3[1], color="orange", zorder=-1)
    plt.savefig('GMM.png')
    plt.close()

    conf_mat = [[0,0,0],[0,0,0],[0,0,0]]
    for i in range(len(x1)):
        sum1 = 0.0
        sum2 = 0.0
        sum3 = 0.0

        for j in range(K):
            sum1 += pi1[j]*mvn.pdf(x1[i], mu1[j], sigma1[j])
        for j in range(K):
            sum2 += pi2[j]*mvn.pdf(x1[i], mu2[j], sigma2[j])
        for j in range(K):
            sum3 += pi3[j]*mvn.pdf(x1[i], mu3[j], sigma3[j])
        
        if sum1 >= sum2 and sum1 >= sum3:
            conf_mat[0][0] += 1
        elif sum2 >= sum3 and sum2 >= sum1:
            conf_mat[0][1] += 1
        else:
            conf_mat[0][2] += 1
    
    for i in range(len(x2)):
        sum1 = 0.0
        sum2 = 0.0
        sum3 = 0.0

        for j in range(K):
            sum1 += pi1[j]*mvn.pdf(x2[i], mu1[j], sigma1[j])
        for j in range(K):
            sum2 += pi2[j]*mvn.pdf(x2[i], mu2[j], sigma2[j])
        for j in range(K):
            sum3 += pi3[j]*mvn.pdf(x2[i], mu3[j], sigma3[j])
        
        if sum1 >= sum2 and sum1 >= sum3:
            conf_mat[1][0] += 1
        elif sum2 >= sum3 and sum2 >= sum1:
            conf_mat[1][1] += 1
        else:
            conf_mat[1][2] += 1
    
    for i in range(len(x3)):
        sum1 = 0.0
        sum2 = 0.0
        sum3 = 0.0

        for j in range(K):
            sum1 += pi1[j]*mvn.pdf(x3[i], mu1[j], sigma1[j])
        for j in range(K):
            sum2 += pi2[j]*mvn.pdf(x3[i], mu2[j], sigma2[j])
        for j in range(K):
            sum3 += pi3[j]*mvn.pdf(x3[i], mu3[j], sigma3[j])
        
        if sum1 >= sum2 and sum1 >= sum3:
            conf_mat[2][0] += 1
        elif sum2 >= sum3 and sum2 >= sum1:
            conf_mat[2][1] += 1
        else:
            conf_mat[2][2] += 1

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

    
    








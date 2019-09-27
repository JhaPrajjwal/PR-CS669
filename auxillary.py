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
        
    else:
        inv = mat_inverse(conv_mat)
        meannp = np.asarray([mu])
        meannpt = np.transpose(meannp)
        x = np.array([v])
        xt = np.transpose(x)
        temp = np.dot(x,inv)
        temp = np.dot(temp, xt)
        val = temp[0][0]
        val1 = -val/2
        # print(inv.shape, meannpt.shape)
        temp = np.dot(inv, np.transpose(meannp))
        temp = np.transpose(temp)
        temp = np.dot(temp, np.transpose(x))
        val = val1 + temp
        val1 = np.dot(meannp, inv)
        val1 = np.dot(val1, meannpt)
        val2 = math.log(np.linalg.det(conv_mat))
        val1 = val1 + val2
        val1 = -val1/2
        val = val + val1
        return val



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

def multivariate_gaussian(pos, mu, Sigma):
    """Return the multivariate Gaussian distribution on array pos.
    pos is an array constructed by packing the meshed arrays of variables
    x_1, x_2, x_3, ..., x_k into its _last_ dimension.
    """
    n = len(mu)
    Sigma_det = det=(Sigma[0][0]*Sigma[1][1])-(Sigma[1][0]*Sigma[0][1])
    Sigma_inv = np.linalg.inv(Sigma)
    N = np.sqrt((2*np.pi)**n * Sigma_det)
    # This einsum call calculates (x-mu)T.Sigma-1.(x-mu) in a vectorized
    # way across all the input variables.
    fac = np.einsum('...k,kl,...l->...', pos-mu, Sigma_inv, pos-mu)
    return np.exp(-fac / 2) / N
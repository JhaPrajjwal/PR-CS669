import math

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



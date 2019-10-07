import matplotlib.pyplot as plt
import math
import data as D
import os

def closest_centroid_cost(K, i, data, centroids):
    min_value = 1000000000000000000000000
    min_index = -1

    for j in range(K):
        val = (data[i][0]-centroids[j][0])**2 + (data[i][1]-centroids[j][1])**2
        val = math.sqrt(val)

        if min_value > val:
            min_value = val
            min_index = j
        
    return min_value, min_index


def assign(K, number_of_points, data, centroids, closest_centroid):
    total_cost = 0.0

    for i in range(number_of_points):
        cost, ind = closest_centroid_cost(K,i, data, centroids)
        closest_centroid[i] = ind
        total_cost += cost
    return total_cost


def update(K, number_of_points, data, centroids, closest_centroid):
    temp = []
    for i in range(K):
        temp.append([0,0,0])

    for i in range(number_of_points):
        temp[closest_centroid[i]][0] += data[i][0]
        temp[closest_centroid[i]][1] += data[i][1]
        temp[closest_centroid[i]][2] += 1

    for i in range(K):
        if temp[i][2] != 0:
            centroids[i][0] = temp[i][0] / temp[i][2]
            centroids[i][1] = temp[i][1] / temp[i][2]

    total_cost = 0.0

    for i in range(number_of_points):
        val = (data[i][0]-centroids[closest_centroid[i]][0])**2 + (data[i][1]-centroids[closest_centroid[i]][1])**2
        val = math.sqrt(val)
        total_cost += val

    return total_cost

def cluster(K, data, centroids, iter):
    classes = {}
    for i in range(len(data)):
        val, ind = closest_centroid_cost(K, i, data, centroids)

        if ind not in classes:
            classes[ind] = [[],[]]
            classes[ind][0].append(data[i][0])
            classes[ind][1].append(data[i][1])
        else:
            classes[ind][0].append(data[i][0])
            classes[ind][1].append(data[i][1])
    
    plt.figure()
    plt.style.use('ggplot')
    plt.scatter(classes[0][0],classes[0][1],color="blue")
    plt.scatter(classes[1][0],classes[1][1],color="red")
    plt.scatter(classes[2][0],classes[2][1],color="green")

    for i in range(K):
        plt.plot([centroids[i][0]], [centroids[i][1]], marker='o', markersize=10, color="black")
    plt.ylabel('Y')
    plt.xlabel('X')
    if iter != -1:
        plt.savefig('plots/iteration'+str(iter)+'.png')
    else:
        plt.savefig('plots/test_data.png')


def kmeans(K, iterations, number_of_points, data, centroids, closest_centroid, test):
    E_step = []
    M_step = []
    X_axis = []

    for i in range(iterations):
        E_step.append(assign(K, number_of_points, data, centroids, closest_centroid))
        M_step.append(update(K, number_of_points, data, centroids, closest_centroid))
        X_axis.append(i+1)
        # print(centroids)
        # print(closest_centroid)
        cluster(K, test, centroids, i)

    return E_step, M_step, X_axis

def metrics(K, data, label, centroids):
    conf_mat = []
    for i in range(K):
        conf_mat.append([0,0,0])

    for i in range(len(data)):
        val, ind = closest_centroid_cost(K, i, data, centroids)
        conf_mat[label[i]][(ind+1)%3] += 1

    for i in range(K):
        for j in range(K):
            print(conf_mat[i][j],end=" ")
        print()

    cnt = 0
    neg = 0
    for i in range(K):
        for j in range(K):
            cnt += conf_mat[i][j]
            if i != j:
                neg += conf_mat[i][j]
    print("Accuracy: ", ((cnt-neg)/float(cnt))*100)


if __name__ == '__main__':

    data, test_data, test_label = D.get_data("./Data1/")
    K = 3
    iterations = 10
    number_of_points = len(data)

    centroids = data[:K]
    closest_centroid = [0] * number_of_points

    if not os.path.isdir("plots"):
        os.mkdir("plots")
        
    E, M, X = kmeans(K, iterations, number_of_points, data, centroids, closest_centroid, test_data)
    plt.figure()
    # plt.plot(X,E,color='red')
    plt.plot(X,M,color='green')
    plt.savefig('plots/cost.png')
    
    cluster(K, test_data, centroids, -1)
    metrics(K, test_data, test_label, centroids)




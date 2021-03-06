import auxillary as Ax
import data
import matplotlib.pyplot as plt
import numpy as np

class1 = [[],[]]
class2 = [[],[]]
class3 = [[],[]]

conf_mat = [[0,0,0],[0,0,0],[0,0,0]]

def decision_boundary(val,mu1,mu2,mu3,cov_mat1,cov_mat2,cov_mat3):

    points_x = [[],[],[]]
    points_y = [[],[],[]]

    start = float(range_x[0])
    while start < float(range_x[1]):
        start_j = float(range_y[0])
        while start_j < float(range_y[1]):
            out = Ax.classify([start, start_j], mu1, mu2, mu3, cov_mat1,cov_mat2,cov_mat3)
            points_x[out].append(start)
            points_y[out].append(start_j)
            start_j += val
        start += val

    plt.scatter(points_x[0],points_y[0],color="red",zorder=-1)
    plt.scatter(points_x[1],points_y[1],color="blue",zorder=-1)
    plt.scatter(points_x[2],points_y[2],color="green",zorder=-1)



print("Welcome. Enter the required option.")
print("1. Linearly Separable")
print("2. Non-Linearly Separable")
print("3. Real World Data")
choice = int(input())

if choice==1:
    # Linearly Separable
    range_x = [-20,25]
    range_y = [-20,20]
    step = 0.25
    file = "./Data1/"
elif choice==2:
    # Non-Linearly Separable
    range_x = [-5,5]
    range_y = [-5,5]
    step = 0.05
    file = "./Data2/"
else:
    # Real Data
    range_x = [-1000,2500]
    range_y = [0,3000]
    step = 20
    file = "./Data3/"
    
   
train1, test1 = data.get_data(file +"Class1.txt")
train2, test2 = data.get_data(file +"Class2.txt")
train3, test3 = data.get_data(file +"Class3.txt")

mu1 = Ax.mean(train1)
mu2 = Ax.mean(train2)
mu3 = Ax.mean(train3)

cov_mat1 = Ax.covariance_mat(train1)
cov_mat2 = Ax.covariance_mat(train2)
cov_mat3 = Ax.covariance_mat(train3)

# cov_mat = [[0,0],[0,0]]

for i in range(2):
    for j in range(2):
        if i!=j:
            cov_mat1[i][j] = 0
            cov_mat2[i][j] = 0
            cov_mat3[i][j] = 0

for i in range(len(test1[0])):
    out = Ax.classify([test1[0][i],test1[1][i]],mu1,mu2,mu3,cov_mat1,cov_mat2,cov_mat3)
    conf_mat[0][out] += 1
    if out == 0:
        class1[0].append(test1[0][i])
        class1[1].append(test1[1][i])
    elif out == 1:
        class2[0].append(test1[0][i])
        class2[1].append(test1[1][i])
    else:
        class3[0].append(test1[0][i])
        class3[1].append(test1[1][i])

for i in range(len(test2[0])):
    out = Ax.classify([test2[0][i],test2[1][i]],mu1,mu2,mu3,cov_mat1,cov_mat2,cov_mat3)
    conf_mat[1][out] += 1
    if out == 0:
        class1[0].append(test2[0][i])
        class1[1].append(test2[1][i])
    elif out == 1:
        class2[0].append(test2[0][i])
        class2[1].append(test2[1][i])
    else:
        class3[0].append(test2[0][i])
        class3[1].append(test2[1][i])

for i in range(len(test3[0])):
    out = Ax.classify([test3[0][i],test3[1][i]],mu1,mu2,mu3,cov_mat1,cov_mat2,cov_mat3)
    conf_mat[2][out] += 1
    if out == 0:
        class1[0].append(test3[0][i])
        class1[1].append(test3[1][i])
    elif out == 1:
        class2[0].append(test3[0][i])
        class2[1].append(test3[1][i])
    else:
        class3[0].append(test3[0][i])
        class3[1].append(test3[1][i])

Ax.call_metric(conf_mat)

plt.figure()
plt.scatter(class1[0],class1[1],color="yellow")
plt.scatter(class2[0],class2[1],color="white")
plt.scatter(class3[0],class3[1],color="red")
Ax.plot_contour(mu1,cov_mat1,test1[0],test1[1],plt)
Ax.plot_contour(mu2,cov_mat2,test2[0],test2[1],plt)
Ax.plot_contour(mu3,cov_mat3,test3[0],test3[1],plt)
decision_boundary(step,mu1,mu2,mu3,cov_mat1,cov_mat2,cov_mat3)
plt.xlim(range_x[0],range_x[1])
plt.ylim(range_y[0],range_y[1])
plt.gca().set_aspect('equal', adjustable='box')
plt.ylabel('Y')
plt.xlabel('X')
plt.show()


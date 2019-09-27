import auxillary as Ax
import data
import matplotlib.pyplot as plt
import numpy as np

class1 = [[],[]]
class2 = [[],[]]
class3 = [[],[]]
range_x = [-20,25]
range_y = [-20,20]

def classify(v, mu1, mu2, mu3, var):
    w1 = Ax.discriminant_func([v[0],v[1]], mu1, var)
    w2 = Ax.discriminant_func([v[0],v[1]], mu2, var)
    w3 = Ax.discriminant_func([v[0],v[1]], mu3, var)

    if w1 >= w2 and w1 >= w3:
        return 0
    elif w2 >= w3 and w2 >= w1:
        return 1
    else:
        return 2


def plot_contour(mu,var,x,y):
    min_x = 100000000
    min_y = 100000000
    max_x = -100000000
    max_y = -100000000
    N = len(x)
    for i in range(len(x)):
        min_x=min(min_x,x[i])
        min_y=min(min_y,y[i])
        max_x=max(max_x,x[i])
        max_y=max(max_y,y[i])


    X=np.linspace(min_x,max_x,N)
    Y=np.linspace(min_y,max_y,N)
    X, Y = np.meshgrid(X, Y)

    Z = []
    for i in range(len(X)):
        temp = []
        for j in range(len(X[0])):
            temp.append(Ax.discriminant_func([X[i][j],Y[i][j]],mu,var))
        Z.append(temp)
        
    plt.contour(X, Y, Z,zorder=100,alpha=0.5,colors=['black'])



def decision_boundary(val,mu1,mu2,mu3,avg_cov):

    points_x = [[],[],[]]
    points_y = [[],[],[]]

    start = float(range_x[0])
    while start < float(range_x[1]):
        start_j = float(range_y[0])
        while start_j < float(range_y[1]):
            out = classify([start, start_j], mu1, mu2, mu3, avg_cov)
            points_x[out].append(start)
            points_y[out].append(start_j)
            start_j += val
        start += val

    plt.scatter(points_x[0],points_y[0],color="red",zorder=-1)
    plt.scatter(points_x[1],points_y[1],color="blue",zorder=-1)
    plt.scatter(points_x[2],points_y[2],color="green",zorder=-1)

            

   
train1, test1 = data.get_data("./Data1/Class1.txt")
train2, test2 = data.get_data("./Data1/Class2.txt")
train3, test3 = data.get_data("./Data1/Class3.txt")


mu1 = Ax.mean(train1)
mu2 = Ax.mean(train2)
mu3 = Ax.mean(train3)

avg_var_1 = Ax.avg_var(train1)
avg_var_2 = Ax.avg_var(train2)
avg_var_3 = Ax.avg_var(train3)

avg_cov = (avg_var_1 + avg_var_2 + avg_var_3)/3

# print(mu1,mu2,mu3)
# print(avg_var_1, avg_var_2, avg_var_3)
# print(avg_cov)

for i in range(len(test1[0])):
    out = classify([test1[0][i],test1[1][i]],mu1,mu2,mu3,avg_cov)
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
    out = classify([test2[0][i],test2[1][i]],mu1,mu2,mu3,avg_cov)
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
    out = classify([test3[0][i],test3[1][i]],mu1,mu2,mu3,avg_cov)
    if out == 0:
        class1[0].append(test3[0][i])
        class1[1].append(test3[1][i])
    elif out == 1:
        class2[0].append(test3[0][i])
        class2[1].append(test3[1][i])
    else:
        class3[0].append(test3[0][i])
        class3[1].append(test3[1][i])

 



plt.figure()
plt.scatter(class1[0],class1[1])
plt.scatter(class2[0],class2[1])
plt.scatter(class3[0],class3[1])
plot_contour(mu1,avg_cov,test1[0],test1[1])
plot_contour(mu2,avg_cov,test2[0],test2[1])
plot_contour(mu3,avg_cov,test3[0],test3[1])
decision_boundary(0.05,mu1,mu2,mu3,avg_cov)
plt.xlim(range_x[0],range_x[1])
plt.ylim(range_y[0],range_y[1])
plt.gca().set_aspect('equal', adjustable='box')
plt.ylabel('Y')
plt.xlabel('X')
plt.show()


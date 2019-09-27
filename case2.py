import auxillary as Ax
import data
import matplotlib.pyplot as plt
import numpy as np

class1 = [[],[]]
class2 = [[],[]]
class3 = [[],[]]

conf_mat = [[0,0,0],[0,0,0],[0,0,0]]

# Real Data
case = 2
range_x = []
range_y = []
step = 0
file = ""

def classify(v, mu1, mu2, mu3, var):
    w1 = Ax.discriminant_func([v[0],v[1]], mu1, conv_mat, case)
    w2 = Ax.discriminant_func([v[0],v[1]], mu2, conv_mat, case)
    w3 = Ax.discriminant_func([v[0],v[1]], mu3, conv_mat, case)

    if w1 >= w2 and w1 >= w3:
        return 0
    elif w2 >= w3 and w2 >= w1:
        return 1
    else:
        return 2


def plot_contour(mu,conv_mat,x,y):
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

    # print(min_x,min_y,max_x,max_y)

    X = np.linspace(min_x,max_x,N)
    Y = np.linspace(min_y,max_y,N)
    X, Y = np.meshgrid(X, Y)

    Z = []
    for i in range(len(X)):
        temp = []
        for j in range(len(X[0])):
            temp.append(Ax.discriminant_func([X[i][j],Y[i][j]],mu,conv_mat,case))
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

print("Welcome. Enter the required option.")
print("1. Linearly Separable")
print("2. Non-Linearly Separable")
print("3. Real World Data")
choice = int(input())

if choice==1:
    # Linearly Separable
    range_x = [-20,25]
    range_y = [-20,20]
    step = 0.05
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
    step = 10
    file = "./Data3/"
    
   
train1, test1 = data.get_data(file +"Class1.txt")
train2, test2 = data.get_data(file +"Class2.txt")
train3, test3 = data.get_data(file +"Class3.txt")


mu1 = Ax.mean(train1)
mu2 = Ax.mean(train2)
mu3 = Ax.mean(train3)

avg_var_1 = Ax.avg_var(train1)
avg_var_2 = Ax.avg_var(train2)
avg_var_3 = Ax.avg_var(train3)

avg_cov = (avg_var_1 + avg_var_2 + avg_var_3)/3

conv_mat = [[avg_cov,    0    ],
            [  0    , avg_cov ]]

# print(mu1,mu2,mu3)
# print(avg_var_1, avg_var_2, avg_var_3)
# print(avg_cov)

for i in range(len(test1[0])):
    out = classify([test1[0][i],test1[1][i]],mu1,mu2,mu3,avg_cov)
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
    out = classify([test2[0][i],test2[1][i]],mu1,mu2,mu3,avg_cov)
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
    out = classify([test3[0][i],test3[1][i]],mu1,mu2,mu3,avg_cov)
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


for i in range(3):
    for j in range(3):
        print(conf_mat[i][j],end=" ")
    print()

print("Accuracy: ", Ax.accuracy(conf_mat))
print("Precision for 1: ", Ax.precision(conf_mat,0))
print("Precision for 2: ", Ax.precision(conf_mat,1))
print("Precision for 3: ", Ax.precision(conf_mat,2))
print("Recall for 1: ", Ax.recall(conf_mat,0))
print("Recall for 2: ", Ax.recall(conf_mat,1))
print("Recall for 3: ", Ax.recall(conf_mat,2))
print("Mean Recall: ", (Ax.recall(conf_mat,0)+Ax.recall(conf_mat,1)+Ax.recall(conf_mat,2))/3)
print("F-Score for 1: ", Ax.f_score(conf_mat,0))
print("F-Score for 2: ", Ax.f_score(conf_mat,1))
print("F-Score for 3: ", Ax.f_score(conf_mat,2))
print("Mean F Score: ", (Ax.f_score(conf_mat,0)+Ax.f_score(conf_mat,1)+ Ax.f_score(conf_mat,2))/3)

plt.figure()
plt.scatter(class1[0],class1[1])
plt.scatter(class2[0],class2[1])
plt.scatter(class3[0],class3[1])
plot_contour(mu1,avg_cov,test1[0],test1[1])
plot_contour(mu2,avg_cov,test2[0],test2[1])
plot_contour(mu3,avg_cov,test3[0],test3[1])
decision_boundary(step,mu1,mu2,mu3,avg_cov)
plt.xlim(range_x[0],range_x[1])
plt.ylim(range_y[0],range_y[1])
plt.gca().set_aspect('equal', adjustable='box')
plt.ylabel('Y')
plt.xlabel('X')
plt.show()


import auxillary as Ax
import data
import matplotlib.pyplot as plt






train1, test1 = data.get_data("./Data1/Class1.txt")
train2, test2 = data.get_data("./Data1/Class2.txt")
# train3, test3 = data.get_data("./Data1/Class3.txt")

avg_var_1 = Ax.avg_var(train1)
avg_var_2 = Ax.avg_var(train2)
# print(avg_var_1, avg_var_2)
# avg_var_3 = Ax.avg_var(train3)

# avg_cov = (avg_var_1 + avg_var_2 + avg_var_3)/3

# cov_mat = [[avg_cov,    0   ],
#            [   0   , avg_cov]]


avg_cov = (avg_var_1 + avg_var_2)/2
class1 = [[],[]]
class2 = [[],[]]
mu1 = Ax.mean(train1)
mu2 = Ax.mean(train2)
# print(mu1,mu2)

for i in range(len(test1[0])):
    w1 = Ax.discriminant_func([test1[0][i],test1[1][i]], mu1, avg_cov)
    w2 = Ax.discriminant_func([test1[0][i],test1[1][i]], mu2, avg_cov)

    if w1 >= w2:
        class1[0].append(test1[0][i])
        class1[1].append(test1[1][i])
    else:
        class2[0].append(test1[0][i])
        class2[1].append(test1[1][i])

for i in range(len(test2[0])):
    w1 = Ax.discriminant_func([test2[0][i],test2[1][i]], mu1, avg_cov)
    w2 = Ax.discriminant_func([test2[0][i],test2[1][i]], mu2, avg_cov)

    if w1 >= w2:
        class1[0].append(test2[0][i])
        class1[1].append(test2[1][i])
    else:
        class2[0].append(test2[0][i])
        class2[1].append(test2[1][i])


print(class1)
print(class2)
plt.scatter(class1[0],class1[1])
plt.scatter(class2[0],class2[1])
# plt.scatter(test1[0],test1[1])
# plt.scatter(test2[0],test2[1])
plt.ylabel('Y')
plt.xlabel('X')
plt.show()

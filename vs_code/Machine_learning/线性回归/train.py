# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from linear_regression import *

##############################画出线性回归直线#####################################
def show_data(x, y, w=None, b=None):
    plt.scatter(x, y, marker='.')
    if w is not None and b is not None:
        plt.plot(x, w*x+b, c='red')
    plt.show()

################################data load###########################################
# data generation
np.random.seed(272)                                                     #随机数种子
data_size = 100                                                         #自变量和应变量数量
x = np.random.uniform(low=1.0, high=10.0, size=data_size)               #自变量
y = x * 20 + 10 + np.random.normal(loc=0.0, scale=10.0, size=data_size) #应变量,分布较为集中

############################画出数据的散点图#########################################
plt.scatter(x, y, marker='.')
plt.show()

############################ train / test split########################################
shuffled_index = np.random.permutation(data_size)                        #对0-100之间的序列进行随机排序
x = x[shuffled_index]
y = y[shuffled_index]
split_index = int(data_size * 0.7)                                        #按位置进行前后划分数据
x_train = x[:split_index]
y_train = y[:split_index]
x_test = x[split_index:]
y_test = y[split_index:]

# visualize data
# plt.scatter(x_train, y_train, marker='.')
# plt.show()
# plt.scatter(x_test, y_test, marker='.')
# plt.show()

###########################train the liner regression model########################
#seed设置数字时，则每次产生的数据是一样的，设置为None时，每次产生的数据时不一样的,max_iter为梯度下降的次数
regr = LinerRegression(learning_rate=0.01, max_iter=10,seed=211)
regr.fit(x_train, y_train)
print('cost: \t{:.3}'.format(regr.loss()))
print('w: \t{:.3}'.format(regr.w))
print('b: \t{:.3}'.format(regr.b))
show_data(x, y, regr.w, regr.b)

# plot the evolution of cost
plt.scatter(np.arange(len(regr.loss_arr)), regr.loss_arr, marker='o', c='green')
plt.show()
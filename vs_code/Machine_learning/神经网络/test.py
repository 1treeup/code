
# _*_ coding: utf-8 _*_

import numpy as np
from sklearn.datasets import load_digits #导入数据集
from sklearn.metrics import confusion_matrix,classification_report   #对结果的预测的包
from sklearn.preprocessing import LabelBinarizer  #把数据转化为二维的数字类型
from sklearn.model_selection import train_test_split   #可以把数据拆分成训练集与数据集


def tanh(x):
    return np.tanh(x)


def tanh_derivative(x):
    return 1 -  np.tanh(x) * np.tanh(x)

# sigmod函数
def logistic(x):
    return 1 / (1 + np.exp(-x))

# sigmod函数的导数
def logistic_derivative(x):
    return logistic(x) * (1 - logistic(x))


class NeuralNetwork:
    def __init__ (self, layers, activation = 'tanh'):
        if activation == 'logistic':
            self.activation = logistic
            self.activation_deriv = logistic_derivative
        elif activation == 'tanh':
            self.activation = tanh
            self.activation_deriv = tanh_derivative

        # 随机产生权重值
        self.weights = []
        for i in range(1, len(layers) - 1):     # 不算输入层，循环
            self.weights.append((2 * np.random.random( (layers[i-1] + 1, layers[i] + 1)) - 1) * 0.25 )
            self.weights.append((2 * np.random.random( (layers[i] + 1, layers[i+1])) - 1) * 0.25 )
            #print self.weights

    def fit(self, x, y, learning_rate=0.2, epochs=10000):
        #创建并初始化要使用的变量
        x = np.atleast_2d(x)#转化X为np数据类型，试数据类型至少是两维的
        temp = np.ones([x.shape[0], x.shape[1]+1])
        temp[:, 0:-1] = x
        x = temp
        y = np.array(y)

        for k in range(epochs): # 循环epochs次
            i = np.random.randint(x.shape[0])   # 随机产生一个数，对应行号，即数据集编号
            a = [x[i]]  # 抽出这行的数据集

            # 迭代将输出数据更新在a的最后一行
            for l in range(len(self.weights)):#完成正向所有的更新
                a.append(self.activation(np.dot(a[l], self.weights[l])))#dot():对应位相乘后相加

            # 减去最后更新的数据，得到误差
            error = y[i] - a[-1]
            deltas = [error * self.activation_deriv(a[-1])]

            # 求梯度
            for l in range(len(a) - 2, 0, -1):
                deltas.append(deltas[-1].dot(self.weights[l].T) * self.activation_deriv(a[l]) )

            #反向排序
            deltas.reverse()

           # 梯度下降法更新权值
            for i in range(len(self.weights)):
                layer = np.atleast_2d(a[i])
                delta = np.atleast_2d(deltas[i])
                self.weights[i] += learning_rate * layer.T.dot(delta)

    def predict(self, x):
        x = np.array(x)
        temp = np.ones(x.shape[0] + 1)
        temp[0:-1] = x
        a = temp
        for l in range(0, len(self.weights)):
            a = self.activation(np.dot(a, self.weights[l]))
        return a

digits = load_digits()  #把数据改成0到1之间
X = digits.data
y = digits.target
X -= X.min()
X /= X.max()

nn = NeuralNetwork([64,100,10],'logistic')
X_train,X_test,y_train,y_test = train_test_split(X,y)
print(y_train)
labels_train = LabelBinarizer().fit_transform(y_train)#把数据转化为二维的数字类型

labels_test = LabelBinarizer().fit_transform(y_test)
print("start fitting")
nn.fit(X_train,labels_train,epochs=3000)
predictions = []
for i in range(X_test.shape[0]):
    o = nn.predict(X_test[i])
    predictions.append(np.argmax(o))
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))
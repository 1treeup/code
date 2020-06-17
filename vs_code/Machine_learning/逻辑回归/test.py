import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV,LinearRegression
import matplotlib.pyplot as plt
import matplotlib as mpl
###############################数据导入喝处理###################################
names=['number1','number2','class']
data=pd.read_csv("../逻辑回归/data.csv",header=None,sep='\s+',names=names)
data=np.array(data)
x=data[:,:2]
y=data[:,2:]
################################分割训练集喝测试集##############################
X_train,X_test,Y_train,Y_test = train_test_split(x,y,test_size=0.3,random_state=0)
#
# #对数据的训练集进行标准化
ss = StandardScaler()
X_train = ss.fit_transform(X_train)     #先拟合数据在进行标准化
#
# ################构建模型###################
#
lr = LogisticRegressionCV(multi_class="ovr",fit_intercept=True,Cs=np.logspace(-2,2,20),cv=2,penalty="l2",solver="lbfgs",tol=0.01)
re = lr.fit(X_train,Y_train)
r = re.score(X_test,Y_test)
print("R值(准确率):",r)
print("参数:",re.coef_)
print("截距:",re.intercept_)

#########################3##########对数据进行分类，便于画图#######################
x0=[]
x1=[]
for i,j,k in data:
    if k==0:
     x0.append([i,j,k])
    else:
     x1.append([i,j,k])
x0=np.array(x0)
x1=np.array(x1)
#####################################画出数据的散点图##############################
# print(x0[:,:1])
# print(x0[:,1:2])
Y_pred = re.predict(X_train)
plt.scatter(x0[:,:1], x0[:,1:2], color='red', marker='o',label='y=0' )
plt.scatter(x1[:,:1], x1[:,1:2], color='blue', marker='x',label='y=1')
plt.legend(loc=2)
# plt.show()

################################画出逻辑回归线#####################################
x = np.arange(-3.0, 3.0, 0.1)
y = (re.intercept_-re.coef_[:,1:]*x)/re.coef_[:,:1]
y = y.reshape(x.shape)
plt.plot(x,y)

plt.show()


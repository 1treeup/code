import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
#import matplotlib as plt改成上面的库形式
adv_data=pd.read_csv("../线性回归/Advertising.csv")
new_adv_data = adv_data.iloc[:,0:4]
#得到我们所需要的数据集且查看其前几列以及数据形状
print(new_adv_data.head(),'\nShape:',new_adv_data.shape)
#特征属性输出
feat_cols = new_adv_data.columns.tolist()[0:]
print(feat_cols)

print(new_adv_data.describe())
#缺失值检验
print(new_adv_data[new_adv_data.isnull()==True].count())
#利用箱图来从可视化方面来查看数据集
new_adv_data.boxplot()

plt.show()
##相关系数矩阵 r(相关系数) = x和y的协方差/(x的标准差*y的标准差) == cov（x,y）/σx*σy
#相关系数0~0.3弱相关0.3~0.6中等程度相关0.6~1强相关
print(new_adv_data.corr())

# # 通过加入一个参数kind='reg'，seaborn可以添加一条最佳拟合直线和95%的置信带。
# #sns.pairplot多变量图函数
sns.pairplot(new_adv_data, x_vars=['TV','radio','newspaper'], y_vars='sales', size=7, aspect=0.8,kind = 'reg')
#
plt.show()
#分割训练集和测试集
X_train,X_test,Y_train,Y_test = train_test_split(new_adv_data.iloc[:,:3],new_adv_data.sales,train_size=.80)
#
# print("原始数据特征:",new_adv_data.iloc[:,:3].shape,
#       ",训练数据特征:",X_train.shape,
#       ",测试数据特征:",X_test.shape)
#
# print("原始数据标签:",new_adv_data.sales.shape,
#       ",训练数据标签:",Y_train.shape,
#       ",测试数据标签:",Y_test.shape)

#引入线性模型，并进行训练，得到参数
model = LinearRegression()
model.fit(X_train,Y_train)
a  = model.intercept_#截距
b = model.coef_#回归系数
print("最佳拟合线:截距",a,",回归系数：",b)
#
# #R方检测
# #决定系数r平方
# #对于评估模型的精确度
# #y误差平方和 = Σ(y实际值 - y预测值)^2
# #y的总波动 = Σ(y实际值 - y平均值)^2
# #有多少百分比的y波动没有被回归拟合线所描述 = SSE/总波动
# #有多少百分比的y波动被回归线描述 = 1 - SSE/总波动 = 决定系数R平方
# #对于决定系数R平方来说1） 回归线拟合程度：有多少百分比的y波动刻印有回归线来描述(x的波动变化)
# #2）值大小：R平方越高，回归模型越精确(取值范围0~1)，1无误差，0无法完成拟合
# score = model.score(X_test,Y_test)
#
# print(score)
#
# #对线性回归进行预测
#
# Y_pred = model.predict(X_test)
#
# print(Y_pred)
#
# plt.plot(range(len(Y_pred)),Y_pred,'b',label="predict")
# #显示图像
# plt.show()
#
#
# plt.figure()
# plt.plot(range(len(Y_pred)),Y_pred,'b',label="predict")
# plt.plot(range(len(Y_pred)),Y_test,'r',label="test")
# plt.legend(loc="upper right") #显示图中的标签
# plt.xlabel("the number of sales")
# plt.ylabel('value of sales')
# plt.show()

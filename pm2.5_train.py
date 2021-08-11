import numpy as np
import pandas as pd
import math

data = pd.read_csv('./train.csv', encoding='big5')

data = data.iloc[:, 3:]
data[data == 'NR'] = 0
data = data.to_numpy()

# 整理数据，把18个特征提取出来
# 5760=12*20*24
# 12个月、每个月20天、每天24小时
data_processed = np.empty((18, 5760))
for i in range(240):
    data_processed[:, 24*i:24*i+24] = data[18*i:18*i+18, :]

# 用前9小时数据预测第10小时的PM2.5指数,x为数据，y为标签
# 每9个小时为一组数据，一个月471组，共471*12组数据
# 每组数据中包含18*9个特征：
# 每小时18个特征*9小时
x = np.empty((471*12, 18*9))
# 每组数据对应1个标签，即第10小时的PM2.5
y = np.empty((471*12, 1))

# 共12个月，用i控制
# 每个月471组数据，用j控制
# 到达新的一月时，从当月第一天开始
for i in range(12):
    for j in range(471):
        x[i*471+j, :] = data_processed[:, i*480+j:i*480+j+9].reshape((1, -1))       # 遍历x的每一行，填入前9天的数据
        y[i*471+j, :] = data_processed[9, i*480+j+9]        # 遍历y的每一行，填入第10天的PM2.5值

x_mean = np.mean(x, axis=0)
x_std = np.std(x, axis=0)

# 标准化，减均值再除以标准差
for i in range(471*12):
    for j in range(18*9):
        x[i][j] = (x[i][j] - x_mean[j]) / x_std[j]

# 随机打乱
np.random.seed(116)
np.random.shuffle(x)
np.random.seed(116)
np.random.shuffle(y)

# 8：2划分训练集与验证集
x_train = x[:math.floor(len(x)*0.8), :]
y_train = y[:math.floor(len(y)*0.8), :]

x_val = x[math.floor(len(x)*0.8):, :]
y_val = y[math.floor(len(x)*0.8):, :]

# 随机初始化w和b,服从正态分布
np.random.seed(17)
w = np.random.normal(size=[18*9, 1])
np.random.seed(17)
b = np.random.normal(size=1)

# 迭代次数
iterations = 7000
# 使用Adagrad算法优化
adagrad = np.zeros((18*9, 1))
adagrad_b = 0
# 初始学习率
lr = 100
# 防止Adagrad分母为0
eps = 10**-10

for i in range(iterations):
    # 预测值
    y_hat = np.dot(x_train, w) + b
    # MSE 损失函数
    loss = np.sum(np.power(y_hat - y_train, 2)) / len(x_train)
    if i % 100 == 0:        # 每100轮使用验证集验证
        y_pre = np.dot(x_val, w) + b
        loss_val = np.sum(np.power(y_pre - y_val, 2)) / len(x_val)
        print('after '+str(i)+' epoch, the validation loss is:', loss_val, ', the train loss is:', loss)
    # 梯度下降更新w和b
    gradient = 2 * np.dot(x_train.T, y_hat-y_train)
    gradient_b = np.sum(2 * (y_hat - y_train))/len(x_train)
    adagrad += gradient ** 2
    adagrad_b += gradient_b ** 2
    w = w - lr * gradient / np.sqrt(adagrad+eps)
    b = b - lr * gradient_b / np.sqrt(adagrad_b+eps)

# 保存训练好的参数
np.save('weights.npy', w)
np.save('bias.npy', b)

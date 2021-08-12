import numpy as np
import pandas as pd
import math


def adagrad(train_set, val_set, train_label, val_label, weights, bias, iterations, lr):
    """梯度下降优化器 Adagrad

        在传统梯度下降的基础上，考虑二阶微分的影响，在一阶微分上取点估计二阶微分

        Args:
            train_set: 训练集数据
            val_set: 验证集数据
            train_label: 训练集标签
            val_label: 验证集标签
            weights: 初始权重
            bias: 初始偏置
            iterations: 迭代次数
            lr: 初始学习率

        Returns:
            返回训练好的参数

            weights: 训练好的权重
            bias: 训练好的偏置
        """
    adagrad_w = np.zeros((18 * 9, 1))
    adagrad_b = 0
    # 防止Adagrad优化时分母为0
    eps = 10 ** -10
    for t in range(iterations):
        # 预测值
        y_hat = np.dot(train_set, weights) + bias

        # MSE 损失函数
        loss = np.sum(np.power(y_hat - train_label, 2)) / len(train_set)
        if t % 100 == 0:  # 每100轮使用验证集验证
            y_pre = np.dot(val_set, weights) + bias
            loss_val = np.sum(np.power(y_pre - val_label, 2)) / len(val_set)
            print('after ' + str(t) + ' epoch, the validation loss is:', loss_val, ', the train loss is:', loss)

        # 计算梯度
        gradient_w = 2 * np.dot(train_set.T, y_hat - train_label) / len(train_set)
        gradient_b = np.sum(2 * (y_hat - train_label)) / len(train_set)

        # Adagrad算法更新w和b
        adagrad_w += gradient_w ** 2
        adagrad_b += gradient_b ** 2
        weights -= lr * gradient_w / np.sqrt(adagrad_w + eps)
        bias -= lr * gradient_b / np.sqrt(adagrad_b + eps)

    return weights, bias


def sgdm(train_set, val_set, train_label, val_label, weights, bias, iterations, beta):
    """梯度下降优化器 SGDM

        带有动量的SGD优化器，每次更新考虑当前权重和上一次更新

        Args:
            train_set: 训练集数据
            val_set: 验证集数据
            train_label: 训练集标签
            val_label: 验证集标签
            weights: 初始权重
            bias: 初始偏置
            iterations: 迭代次数
            beta: 超参，控制当前权重和上次更新在本次更新所占比重

        Returns:
            返回训练好的参数

            weights: 训练好的权重
            bias: 训练好的偏置
        """
    vw = np.zeros((18 * 9, 1))      # 动量
    vb = np.zeros(1)

    for t in range(iterations):
        # 预测值
        y_hat = np.dot(train_set, weights) + bias

        # MSE 损失函数
        loss = np.sum(np.power(y_hat - train_label, 2)) / len(train_set)
        if t % 100 == 0:  # 每100轮使用验证集验证
            y_pre = np.dot(val_set, weights) + bias
            loss_val = np.sum(np.power(y_pre - val_label, 2)) / len(val_set)
            print('after ' + str(t) + ' epoch, the validation loss is:', loss_val, ', the train loss is:', loss)

        # 计算梯度
        gradient_w = 2 * np.dot(train_set.T, y_hat - train_label) / len(train_set)
        gradient_b = np.sum(2 * (y_hat - train_label)) / len(train_set)

        # SGDM算法优化
        vw = beta * vw + (1 - beta) * gradient_w
        vb = beta * vb + (1 - beta) * gradient_b
        weights -= vw
        bias -= vb

    return weights, bias


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

iteration_time = 7000       # 迭代次数
learning_rate = 100     # Adagrad 初始学习率
beta = 0.99     # SGDM 超参
w, b = sgdm(x_train, x_val, y_train, y_val, w, b, iteration_time, beta)

# 保存训练好的参数
np.save('weights.npy', w)
np.save('bias.npy', b)

import numpy as np
import pandas as pd
import csv

data = pd.read_csv("./test.csv", encoding='big5', header=None)

data = data.iloc[:, 2:]
data[data == 'NR'] = 0
data = data.to_numpy()

x = np.empty((240, 18*9))

# 调整数据维度
for i in range(240):
    x[i, :] = data[18*i:18*(i+1), :].reshape((1, -1))

# 标准化
x_mean = np.mean(x, axis=0)
x_std = np.std(x, axis=0)
for i in range(240):
    for j in range(18*9):
        x[i][j] = (x[i][j] - x_mean[j]) / x_std[j]

w = np.load('./weights.npy')
b = np.load('./bias.npy')

y = np.dot(x, w) + b

# 将结果写入csv文件
with open('result.csv', mode='w', newline='') as submit_file:
    csv_writer = csv.writer(submit_file)
    header = ['id', 'value']
    print(header)
    csv_writer.writerow(header)
    for i in range(240):
        row = ['id_' + str(i), y[i][0]]
        csv_writer.writerow(row)
        print(row)

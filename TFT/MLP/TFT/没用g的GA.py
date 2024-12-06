import numpy
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import torch.utils.data as Data
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

data = pd.read_excel('E:/pycharm/my_code/TFT/拼接计划/拼接计划（无8且仅Id）.xlsx', engine='openpyxl')
all_input = data[['Vg', 'Vd', 'W/L']].values
all_output = data[['Id1']].values
"""
    这里，z 是一个 StandardScaler 对象，用于标准化数据。
    StandardScaler 通过去除均值并缩放到单位方差来标准化特征。
"""
z = preprocessing.StandardScaler()
# all_output=minmax.fit_transform(data['Idrain'].values.reshape(-1,1))

(x_train, x_test, y_train, y_test) = train_test_split(all_input, all_output, train_size=0.8,
                                                      random_state=0)
a = np.std(x_train, axis=0)
print(a)
# 标准化处理
x_train = z.fit_transform(x_train)
x_test = z.transform(x_test)
# z.inverse_transform


# 数据类型处理
train_xt = torch.from_numpy(x_train.astype(np.float32))
train_yt = torch.from_numpy(y_train.astype(np.float32))
test_xt = torch.from_numpy(x_test.astype(np.float32))
test_yt = torch.from_numpy(y_test.astype(np.float32))

# 将数据处理为数据加载器
train_data = Data.TensorDataset(train_xt, train_yt.float())
test_data = Data.TensorDataset(test_xt, test_yt.float())

train_loader = Data.DataLoader(dataset=train_data, batch_size=32, shuffle=True, num_workers=0)
test_loader = Data.DataLoader(dataset=test_data, batch_size=32, shuffle=False, num_workers=0)


class MLPregression(nn.Module):
    def __init__(self, p):
        super(MLPregression, self).__init__()
        self.p = torch.from_numpy(p)
        self.p = self.p.to(torch.float32)
        # self.p=p
        self.activate = nn.Sigmoid()
        # 定义第一个隐藏层
        self.input = nn.Linear(in_features=3, out_features=15, bias=True)
        # 定义第二个隐藏层
        self.hidden2 = nn.Linear(15, 8)
        # 回归预测层
        self.predict = nn.Linear(8, 1)
        # 取出权重
        self.input.weight.data = self.p[0:45].reshape((15, 3))
        self.input.bias.data = self.p[45:60].reshape((15,))
        self.hidden2.weight.data = self.p[60:180].reshape((8, 15))
        self.hidden2.bias.data = self.p[180:188].reshape((8,))
        self.predict.weight.data = self.p[188:196].reshape((1, 8))
        self.predict.bias.data = self.p[196:197]

    def forward(self, x):
        x = self.activate(self.input(x))
        x = self.activate(self.hidden2(x))
        output = self.predict(x)
        return output[:, 0]


# mlpreg.load_state_dict(torch.load("C:/Users/XHM/LEVIST/DIST/mlp_init_nosgd.pth"))
def loss_function(id_pre, ids):
    loss_func = nn.MSELoss()
    j = loss_func(id_pre, ids)
    return j


def fitness(x):
    mlpreg = MLPregression(p=x)
    train_loss_all = []
    for step, (b_x, b_y) in enumerate(train_loader):
        output = mlpreg(b_x)
        b_x.requires_grad_(True)
        y = mlpreg(b_x)
        y.retain_grad()
        y.backward(torch.ones_like(y))
        loss = loss_function(output, b_y[:, 0])
    return loss.item()


from sko.GA import GA

ga = GA(func=fitness, n_dim=197, size_pop=200, max_iter=800, lb=[0.001] * 197, ub=[10] * 197, precision=1e-7)
# ga.register(operator_name='selection', operator=selection_tournament, tourn_size=3)
best_x, best_y = ga.run()
print('best_x:', best_x, '\n', 'best_y:', best_y)

torch.save(best_x, "ga.best_x（无8且仅Id）.pth")
Y_history = pd.DataFrame(ga.all_history_Y)
Y_history.min(axis=1).cummin().plot(kind='line')
plt.xlabel("cycles")
plt.xlim((0, 100))
plt.ylabel("fitness")
plt.show()

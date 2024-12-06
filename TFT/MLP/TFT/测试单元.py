import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn import preprocessing
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import torch.utils.data as Data

data=pd.read_excel('E:/pycharm/my_code/TFT/拼接计划/拼接计划.xlsx',engine='openpyxl')
all_input=data[['Vg','Vd','W/L']].values
all_output=data[['Id1','gm1','gd1']].values
z=preprocessing.StandardScaler()
#all_output=minmax.fit_transform(data['Idrain'].values.reshape(-1,1))

(x_train,x_test,y_train,y_test)=train_test_split(all_input,all_output,train_size=0.8,
                                                 random_state=0)
a=np.std(x_train, axis = 0)
print(a)
#标准化处理
x_train=z.fit_transform(x_train)
x_test=z.transform(x_test)

test_xt = torch.from_numpy(x_test.astype(np.float32))
test_yt = torch.from_numpy(y_test.astype(np.float32))

# 将测试数据处理为数据加载器
test_data = Data.TensorDataset(test_xt, test_yt.float())
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

# 加载保存的权重
best_x = torch.load("E:/pycharm/my_code/TFT/MLP/TFT/ga.best_x（无8且仅Id）.pth")

# 创建模型实例并加载权重
mlpreg = MLPregression(p=best_x)

predictions = []
labels = []
with torch.no_grad():
    for step, (b_x, b_y) in enumerate(test_loader):
        output = mlpreg(b_x)
        predictions.extend(output.numpy())
        labels.extend(b_y[:, 0].numpy())

# 计算 R2 分数
r2 = r2_score(labels, predictions)
print(f'R2 Score: {r2}')
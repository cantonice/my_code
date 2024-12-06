import torch
import torchvision.utils
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim

import matplotlib.pyplot as plt
from torchvision.utils import make_grid

import os
import argparse


# 参数设定
parser = argparse.ArgumentParser(description="pytorch_net")  # 用来装载参数的容器
# 给这个解析对象添加命令行参数
parser.add_argument("-f", "--pth_file_dir", type=str, default=None, help="预训练模型相对path")
parser.add_argument("-ep", "--epoch", type=int, default=2, help="训练遍历数据集的过程,需要多少次")
parser.add_argument("-save", "--save_file_pth", type=str, default="my_net.pth", help="保存训练模型名称")
args = parser.parse_args()


# 定义数据集预处理操作
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),  # 数据增强：随机翻转图片
    transforms.RandomCrop(32, padding=4),  # 数据增强：随机裁剪图片
    transforms.ToTensor(),  # 将PIL.Image或者numpy.ndarray数据类型转化为torch.FloatTensor,归一化到【0-1】之间
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))  # 标准化（这里的均值和标准差是CIFAR10数据集的）
])


 # 下载并加载训练数据集
train_set = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
train_loader = DataLoader(train_set, batch_size=64, shuffle=True, num_workers=2)

# 下载并加载测试数据集
test_set = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
test_loader = DataLoader(test_set, batch_size=64, shuffle=False, num_workers=2)


def pth_is_exist(pth_file_dir):
    """
    检查预训练模型是否存在
    :param pth_file_dir: 预训练模型权重文件
    :return: 布尔值真或假
    """
    if os.path.exists(pth_file_dir):
        return True
    else:
        return False

class MyNet(nn.Module):
    """
    定义网络模型，使用两个卷积层和两个全连接层
    """
    def __init__(self):
        super(MyNet,self).__init__()
        # 输入通道3，输出通道6，卷积核大小5,步长默认为1
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5)
        # 池化核大小2， 步长2
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # 输入通道6，输出通道16，卷积核大小5
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        # 全连接层1，池化后输入应该是16*5*5的特征图，在全连接层之前，通常需要将多维的特征图展平（Flatten）成一维向量，以匹配全连接层的输入维度
        self.fc1 = nn.Linear(in_features=16*5*5, out_features=120)
        # 全连接层2
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        # 全连接层3,输出维度为10，因为'CIFAR10'有十类
        self.fc3 = nn.Linear(in_features=84, out_features=10)

    def forward(self,x):
        # 第一层卷积+ReLu激活+池化
        x = self.pool(F.relu(self.conv1(x)))

        # 第二层卷积+ReLu激活+池化
        x = self.pool(F.relu(self.conv2(x)))

        # 将特征图展平,其中-1告诉PyTorch自动计算第一个维度的大小，以便保持张量的元素总数不变
        x = x.view(-1,16*5*5)

        #第一层全连接+ReLu激活函数
        x = F.relu(self.fc1(x))

        # 第二层全链接+ReLu激活函数
        x = F.relu(self.fc2(x))

        # 第三层全连接做输出层
        x = self.fc3(x)
        return x


# 创建网络
my_net = MyNet()

# 定义损失函数
criterion = nn.CrossEntropyLoss()

# 定义优化器
optimizer = optim.SGD(params=my_net.parameters(), lr=0.001, momentum=0.9)  # 动量0.9，意味着先前的梯度将以90%的权重被考虑在内


def main():
    # 训练网络
    # 如果有预训练模型，那就加载一下
    print("加载预训练模型....")
    if pth_is_exist(args.pth_file_dir):
        my_net.load_state_dict(torch.load(args.pth_file_dir))
        print("加载成功！")
    else:
        print("加载失败！预训练模型不存在")

    print("training....")
    for epoch in range(args.epoch):

        running_loss = 0.0
        for i, data in enumerate(train_loader,0):
            """
                这行代码的作用是遍历 train_loader 中的所有批次数据，并为每个批次数据提供一个索引 i。
                在每次迭代中，i 表示当前批次的索引，data 表示当前批次的数据
                0 是 enumerate 函数的第二个参数，表示起始索引值。
            """
            # 获取输入数据
            inputs,labels = data  # data通常是一个包含输入数据和标签的元组，需要将其进一步解构为inputs和labels

            # 梯度清零
            optimizer.zero_grad()

            # 前向传播
            outputs = my_net(inputs)

            # 计算损失
            loss = criterion(outputs, labels)

            # 反向传播
            loss.backward()

            # 更新数据
            optimizer.step()

            # 打印统计信息
            running_loss+=loss.item()  # loss.item()是损失值的Python标量表示
            if i % 100 == 99:
                print("[%d ,%5d] loss: %.3f" %(epoch+1, i+1, running_loss/99))
                running_loss = 0.0
    print("训练完成！")


    # 保存训练模型
    print("保存模型中...")
    torch.save(my_net.state_dict(),args.save_file_pth)
    print("保存成功！")


    # 加载一些测试图片
    dataiter = iter(test_loader)
    images,labels = next(dataiter)

    # 打印图片
    grid = make_grid(images)
    plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
    plt.show()
    # plt.imshow(grid)

    # 显示真实标签
    classes = ["plane", "car", "bird", "cat", "deer", "dog", "frog", "horse'", "ship", "truck"]
    print("GroundTruth:\n","".join("%s\t"% classes[labels[j]]for j in range(len(labels))))


    # 让网络做出预测
    outputs = my_net(images)

    # 预测的标签是最大输出的标签
    _,predicted = torch.max(outputs,1)

    # 显示预测的标签
    print("Predicted:\n","".join("%s\t" % classes[predicted[j]] for j in range(len(predicted))))

    # 测试网络ing
    print("testing....")
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = my_net(images)
            _,predicted = torch.max(outputs.data,1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print("Accuracy of the network on the 10000 test images: %d %%" % (
        100 * correct / total))


if __name__ == '__main__':
    main()
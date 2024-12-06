import matplotlib.pyplot as plt
import numpy as np


def test_sub_polt():
    # 创建x，从-2到2，100个点
    x = np.linspace(-2,2,100)

    # 一次函数
    y1 = x

    # 二次函数
    y2 = x**2

    # 正弦函数
    y3 = np.sin(x)

    # 反比例函数
    y4 = 1/x

    # 创建一张画布
    fig = plt.figure(figsize=(10,8),facecolor="w",dpi=300)

    # 子图1
    ax1 = plt.subplot(221)  #表示2行2列的第1个图
    ax1.set_title("1")
    ax1.plot(x,y1,label = "y1 = x",color = "r")
    # 子图2
    ax2 = plt.subplot(222)  #表示2行2列的第2个图
    ax2.set_title("2")
    ax2.plot(x,y2,label = "y2 = x^2",color = "g")
    # 子图3
    ax3 = plt.subplot(223)
    ax3.set_title("3")
    ax3.plot(x,y3,label = "y3 = sin(x)",color = "b")
    # 子图4
    ax4= plt.subplot(224)
    ax4.set_title("4")
    ax4.plot(x,y4, label = "y4 = 1/x",color = "c")

    fig.tight_layout()  # 自动调整布局
    plt.show()


def test_sub_plots():
    # 1.配置表达式
    x = np.linspace(-2, 2, 100) # 创建x，从-2到2，100个点
    y1 = x  # 一次函数
    y2 = x ** 2  # 二次函数
    y3 = np.sin(x)  # 正弦函数
    y4 = 1 / x  # 反比例函数

    # 2.创建画布
    """
        当使用 `plt.subplots(2, 2)` 这个函数时，你实际上是在创建一个2x2的子图网格。
        这个函数返回两个对象：一个图形对象 `fig` 和一个子图对象的数组 `axes`。   
        这里涉及到Python中的一个特性，称为“解包”（unpacking）。当一个函数返回多个值时
    """
    fig,axes = plt.subplots(2,2)

    # 3.绘制四个子图
    axes[0,0].plot(x,y1,color = "r")
    axes[0,1].plot(x,y2,color = "g")
    axes[1,0].plot(x,y3,color = "b")
    axes[1,1].plot(x,y4,color = "c")

    # 4.设置标题
    axes[0,0].set_title("y1 = x")
    axes[0,1].set_title("y2 = x ** 2")
    axes[1,0].set_title("y3 = np.sin(x)")
    axes[1,1].set_title("y4 = 1 / x")

    # 5.显示网格
    for ax in axes.flat:
        ax.grid(True)

    # 6.自动调整子图布局以适应整个图像
    fig.tight_layout()

    # 7.显示图像
    plt.show()


if __name__ =="__main__":
    test_sub_polt()
    test_sub_plots()
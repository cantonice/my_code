import inline as inline
import matplotlib.pyplot as plt
import numpy as np

# # 当浏览器不显示绘图图片时，加上这一行
# %matplotlib inline
# # 让图片可以显示中文
# plt.rcParams['font.sans-serif'] = 'SimHei'
# # 让图片可以显示负号
# plt.rcParams['axes.unicode_minus'] = False
#
# from matplotlib.font_manager import FontManager
#
# fm = FontManager()
# my_fonts = set(f.name for f in fm.ttflist)
# print(my_fonts)


"""
    --- 画布配置 --- 
    **基本配置**
    =============   =============
        参数	            配置
    =============   =============
      figsize	       画布宽高
        dpi	           分辨率
     facecolor	     画布背景颜色
    =============   =============
"""
plt.figure(figsize=(5,3),dpi=300,facecolor="w")


# 创建x的值，这里我们使用np.linspace来生成100个点
x = np.linspace(0, 2*np.pi, 100)
y1 = np.sin(x)
y2 = np.cos(x)


"""
    --- 绘制函数图像 ---
    **color**
    可以有三种写法，单个字母简写，英文单词，十六进制
    
    **Line Styles**
    =============    ===============================
    character        description
    =============    ===============================
    -          		 solid line style
    --        		 dashed line style
    -.         		 dash-dot line style
    :         		 dotted line style
    =============    ===============================
"""
plt.plot(x, y1, label="y = sin(x)",color = "r", linestyle = "-.")
plt.plot(x, y2, label="y = cos(x)",color = "g", linestyle = "-")

# 添加图例
plt.legend()

# 添加标题和轴标签
plt.title('y = sin(x)/y = cos(x)')
plt.xlabel('x')
plt.ylabel('y1/y2')

# 显示网格
plt.grid(True)

# 显示图像
plt.show()


def draw_ajmd_curve():
    """
    阿基米德螺线绘画
    :return:
    """
    theta = np.arange(0, 6 * np.pi, 0.001 * 6 * np.pi)
    a = 0.1
    b = 10
    r = a + b * theta
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    plt.plot(x, y)
    plt.title('Archimedes Spiral')
    plt.grid(True)
    plt.axis('equal')
    plt.show()
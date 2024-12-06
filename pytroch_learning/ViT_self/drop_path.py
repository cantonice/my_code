import torch.nn as nn
import torch
def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    每个样本的下降路径（随机深度）（应用于残差块的主路径时）。
    这与我为 EfficientNet 等网络创建的 DropConnect impl 相同，但是，
    原名具有误导性，因为“Drop Connect”是另一篇论文中另一种形式的辍学......
    请参阅讨论：https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ...我选择了
    将图层和参数名称更改为“放置路径”，而不是将 DropConnect 混合为图层名称并使用
    以“存活率”为论据。
    """

    if drop_prob == 0. or not training:  # 如果丢弃比例为0或者在非训练模式下，直接返回x
        return x
    # 保留的分支概率
    keep_prob = 1 - drop_prob
    # shape (b, 1, 1, 1)，其中x.ndim输出结果为x的维度，即4。目的是为了创建一个失活矩阵。
    # (1,) * (x.ndim - 1)：这部分代码创建了一个元组，其中包含 x.ndim - 1 个 1
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets

    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    # 向下取整用于确定保存哪些样本，floor_()是floor的原位运算
    random_tensor.floor_()  # binarize
    # 除以keep_drop让一部分分支失活，恒等映射
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


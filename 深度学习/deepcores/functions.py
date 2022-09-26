# coding: utf-8
import numpy as np

#保存深度学习时使用的关键函数
#对输入为二位矩阵、张量均适用

#输入x为一维,返回标量
#输入x为二维,返回一维向量
def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T  #返回的是一维的向量

    x = x - np.max(x) # 溢出对策
    return np.exp(x) / np.sum(np.exp(x))

#输入y,t形状相同，可以是一维，表示1个特征标签。也可以二维，表示batch_size个特征标签
#返回值为标量
def cross_entropy_error(y, t):#返回的结果值是个标量
    if y.ndim == 1:  #如果输入y是一维，需要扩展为二维
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    # t是one-hot-vector的情况下，转换为正确标签所在的索引
    if t.size == y.size:
        t = t.argmax(axis=1)   #获取每一行为1的索引

    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size  #从y中提取同t相同索引的位置的值，计算交叉熵；除以batch_size，表示结果是求取平均每个样本的交叉熵

#输入x为一维，表示1个one-hot-vector类型的特征标签
#输入x为二位，表示batch_size个特征标签
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
# coding: utf-8
import numpy as np
from collections import OrderedDict
from deepcores.layers import *


class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):

        # 初始化权重
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.rand(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.rand(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

        # 生成层
        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['Relu1'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])

        self.lastlayer = SoftmaxWithLoss()

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x

    # x:输入数据, t:监督数据
    def loss(self, x, t):
        y = self.predict(x)
        return self.lastlayer.forward(y, t)

    # x:输入数据, t:监督数据
    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)  # 获取输出结果y中每一行(一行表示一个样本的结果值)中最大值的索引
        if t.ndim != 1:
            t = np.argmax(t, axis=1)  # 获取监督数据t中每一行中最大的索引
        accuracy = np.sum(y == t) / float(x.shape[0])  # 分母表示验证数据的大小
        return accuracy

    # x:输入数据, t:监督数据
    def numerical_gradient(self, x, t):
        pass

    # x:输入数据, t:监督数据
    def gradient(self, x, t):
        self.loss(x, t)  # 触发正向传播
        dout = 1
        dout = self.lastlayer.backward(dout)
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # 设定
        grads = {}
        grads['W1'] = self.layers['Affine1'].dW
        grads['b1'] = self.layers['Affine1'].db
        grads['W2'] = self.layers['Affine2'].dW
        grads['b2'] = self.layers['Affine2'].db

        return grads

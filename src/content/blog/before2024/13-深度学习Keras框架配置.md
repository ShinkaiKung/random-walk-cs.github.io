---
author: 公子小白
pubDatetime: 2020-04-18T13:44:00Z
title: 深度学习Keras框架配置
postSlug: "13"
featured: false
draft: false
tags:
  - "keras"
  - "deep-learning"
description: 深度学习Keras框架配置
---

其他相关文章：[利用Keras对时间序列进行预测]

## Table of Contents

## Keras简介（引自：Keras中文文档）

Keras 是一个用 Python 编写的高级神经网络 API，它能够以 [TensorFlow](https://github.com/tensorflow/tensorflow), [CNTK](https://github.com/Microsoft/cntk), 或者 [Theano](https://github.com/Theano/Theano) 作为后端运行。Keras 的开发重点是支持快速的实验。能够以最小的时延把你的想法转换为实验结果，是做好研究的关键。

如果你在以下情况下需要深度学习库，请使用 Keras：

1. 允许简单而快速的原型设计（由于用户友好，高度模块化，可扩展性）。
2. 同时支持卷积神经网络和循环神经网络，以及两者的组合。
3. 在 CPU 和 GPU 上无缝运行。

**指导原则**

1. 用户友好。 Keras 是为人类而不是为机器设计的 API。它把用户体验放在首要和中心位置。Keras 遵循减少认知困难的最佳实践：它提供一致且简单的 API，将常见用例所需的用户操作数量降至最低，并且在用户错误时提供清晰和可操作的反馈。
2. 模块化。 模型被理解为由独立的、完全可配置的模块构成的序列或图。这些模块可以以尽可能少的限制组装在一起。特别是神经网络层、损失函数、优化器、初始化方法、激活函数、正则化方法，它们都是可以结合起来构建新模型的模块。
3. 易扩展性。 新的模块是很容易添加的（作为新的类和函数），现有的模块已经提供了充足的示例。由于能够轻松地创建可以提高表现力的新模块，Keras 更加适合高级研究。
4. 基于 Python 实现。 Keras 没有特定格式的单独配置文件。模型定义在 Python 代码中，这些代码紧凑，易于调试，并且易于扩展。

## 环境准备

ANACONDA, Windows 8

## 安装TensorFlow

在安装Keras之前，要先安装后端TensorFlow，Theano或者 CNTK。 此处安装TensorFlow。

在cmd窗口中输入以下两行代码：连接到清华镜像网站。

```shell
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --set show_channel_urls yes
```

![](@assets/images/20-1-1024x196.png)

然后输入以下代码：python版本号可在cmd下输入python确定。

```shell
conda create -n tensorflow python=3.7.6
```

输入"y"，继续程序。

![](@assets/images/20-2.png)

激活TensorFlow：

```shell
activate tensorflow
```

![](@assets/images/20-3.png)

然后安装Tensorflow CPU版本：

```shell
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple tensorflow
```

![](@assets/images/20-4-1024x393.png)

打开ANACONDA NAVIGATOR，在tensorflow下（原显示的是base）安装解释器，此处安装Jupyter Notebook。

![](@assets/images/20-5-1024x438.png)

测试TensorFlow是否安装成功，（用于测试TensorFlow 2.0版本）在JuputerNotebook中运行以下代码：

```python
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
hello = tf.constant('hello,tensorflow')
sess= tf.compat.v1.Session()
print(sess.run(hello))
```

无报错输出即可。

## 安装Keras

打开cmd窗口，激活TensorFlow：

```shell
activate tensorflow
```

安装Keras：

```shell
pip install keras
```

![](@assets/images/20-6-1024x348.png)

测试是否安装成功，在python下输入以下代码：

```python
import keras
```

如果输出“Using TensorFlow backend.”则成功。

![](@assets/images/20-7-1024x187.png)

没有GPU的穷人可以忽略下面两行无关输出。。。

至此，Keras就安装完成了！

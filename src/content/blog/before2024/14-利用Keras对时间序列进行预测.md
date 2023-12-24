---
author: 公子小白
pubDatetime: 2020-04-19T10:51:00Z
title: 利用Keras对时间序列进行预测
postSlug: "14"
featured: false
draft: false
tags:
  - "keras"
  - "deep-learning"
description: 利用Keras对时间序列进行预测
---

**注：**本文是笔者在学习**《python深度学习》[美] 弗朗索瓦·肖莱**一书时的学习笔记，文中案例也来源于本书。

其他相关文章：[深度学习Keras框架配置]

## Table of Contents

## 知识准备

### 神经网络的基本概念

- 层(layer)：是一种数据处理模块，可以看成数据过滤器。
- 编译(compile)三要素：
  - 损失函数(loss function)：网络如何衡量在训练数据上的性能，即网络如何朝着正确的方向前进。
  - 优化器(optimazer)：基于训练数据和损失函数来更新网络的机制。
  - 在训练和测试中需要监控的指标(metric)。
- 数据张量（机器学习的基本数据结构）的表示：
  - 向量数据：2D张量，形状为(samples, features)。
  - 时间序列数据或序列数据：3D张量，形状为(samples, timesteps, features)。
  - 图像：4D张量，形状为(samples, height, width, channals)或(samples, channels, height, width)。
  - 视频：5D张量，形状为(samples, frames, height, width, channels)或(samples, frames, channels, height, width)。

### 选择层(layer)的一般原则

- 2D张量：密集连接层[densely connected layer]。
- 3D张量：循环层[recurrent layer，比如LSTM层]。
- 4D张量：二维卷积层[比如Conv2D层]。

### 激活函数(activation)的选择参考

- 中间层选择：activation='relu'
- 输出层选择：activation='sigmoid'（返回0~1之间的值）
- 有时也可以不设定。

### 选择损失函数(loss function)的一般原则

- 二分类问题：二元交叉熵(binary crossentropy)损失函数。
- 多分类问题：分类交叉熵(categorical crossentropy)损失函数。
- 回归问题：均方误差(mean-squraed error)损失函数。
- 序列学习问题：联结主义时序分类(CTC, connectionist temporal classification)损失函数。

### 循环神经网络RNN

循环神经网络，是一类具有内部环的神经网络。RNN处理序列的方式是，遍历所有元素，并保存一个状态，其包含与已查看内容相关的信息。

### Keras可用的循环层

**SimpleRNN层**：过于简化，没有实际价值，存在梯度消失问题。为缓解梯度消失问题，出现了LSTM层和GRU层。

**LSTM层**：是长短期记忆（LSTM, long short-term memory）算法的实现。LSTM层是SimpleRNN层的一种变体，它增加了一种携带信息跨越多个时间步的方法。假设有一条传送带，其运动方向平行于你所处理的序列，序列中 信息可以在任意位置跳上传送带，然后被传送到更晚的时间步，并在需要时原封不动地跳回来——保存信息以便后面使用，从而防止较早期的信号在处理过程中逐渐消失。

**GRU层**：运算原理与LSTM相同，但它做了一些简化，因此运算代价更低。是计算代价和表示能力的折中。

### Keras神经网络的构建和编译说明示例

```python
model = Sequential()    #因为需要将同一个模型多次实例化，所以用一个函数来构建模型。
model.add(layers.GRU(32, input_shape=(None, float_data.shape[-1])))    #选择GRU层作为隐藏层；该层有32个隐藏单元；设置输入数据类型。
model.add(layers.Dense(1))    #选择Dense层作为输出层。
model.compile(optimizer=RMSprop(), loss='mae')    #设置编译参数
history = model.fit_generator(train_gen,    #选择训练集
                              steps_per_epoch=500,    #每期运行500步
                              epochs=20,    #一共运行20期
                              validation_data=val_gen,    #在训练数据上评估模型
                              validation_steps=val_steps)
```

## 温度预测问题——循环神经网络RNN实践

数据来源：[https://s3.amazonaws.com/keras-datasets/jena_climate_2009_2016.csv.zip](https://s3.amazonaws.com/keras-datasets/jena_climate_2009_2016.csv.zip)

需要库：matplotlib, numpy, pandas, keras

目的：根据数据集中每10分钟记录的14个量，包括气温、气压、湿度、风向等，预测24小时之后的气温

### 读入数据

_观测数据：_

```python
import os

data_dir = ''    #数据文件放置在同一文件夹下。
fname = os.path.join(data_dir, 'jena_climate_2009_2016.csv')

f = open(fname)
data = f.read()
f.close()

lines = data.split('\n')
header = lines[0].split(',')
lines = lines[1:]

print(header)    #输出表头
print(len(lines))     #输出行数
```

输出：['"Date Time"', '"p (mbar)"', '"T (degC)"', '"Tpot (K)"', '"Tdew (degC)"', '"rh (%)"', '"VPmax (mbar)"', '"VPact (mbar)"', '"VPdef (mbar)"', '"sh (g/kg)"', '"H2OC (mmol/mol)"', '"rho (g/m\*\*3)"', '"wv (m/s)"', '"max. wv (m/s)"', '"wd (deg)"'] 420551

_解析数据_：将420551行数据换成Numpy数组

```python
import numpy as np

float_data = np.zeros((len(lines), len(header) - 1))
for i, line in enumerate(lines):
    values = [float(x) for x in line.split(',')[1:]]
    float_data[i, :] = values
```

_绘制温度时间序列：_

```python
from matplotlib import pyplot as plt

temp = float_data[:, 1]  # temperature (in degrees Celsius)
plt.plot(range(len(temp)), temp)
plt.show()
```

![](@assets/images/21-1-300x200.png)

_绘制前十天的温度时间数据_：

```python
plt.plot(range(1440), temp[:1440])
plt.show()
```

![](@assets/images/21-2-300x197.png)

### 数据预处理

*数据标准化：*将每个时间序列减去其平均值然后除以其标准差。使用前200000个时间步作为训练数据。

> **注意：**用于测试数据标准化的均值和标准差都是在训练数据上计算得到的，在工作流程中，不能使用在测试数据上计算得到的任何结果，即使是像数据标准化这么简单的事情也不行。

```python
mean = float_data[:200000].mean(axis=0)
float_data -= mean
std = float_data[:200000].std(axis=0)
float_data /= std
```

问题的确切表述：一个时间步是10分钟，每steps个时间步采样一次数据，给定过去lookback个时间步之内的数据，能否预测delay个时间步之后的温度?

*生产时间序列样本及其目标的生成器：*生成器参数如下

- data：浮点数数据组成的原始数组，已经标准化。
- lookback：输入数据应该包括过去多少个时间步。
- delay：目标应该在未来多少个时间步之后。
- min_index和max_index：data数组中的索引，用于界定需要抽取哪些时间步。这有助于保存一部分数据用于验证、另一部分用于测试。
- shuffle：是打乱样本还是顺序抽取数据。
- batch_size：每个批量的样本数。
- step：数据采集的周期（单位：时间步）。我们将其设为6，为的是每小时抽取一个数据点。（简化数据集）

> **为何要使用generator？**
>
> 因为数据集中的样本是高度冗余的，对于第N个样本和第N+1个样本，大部分时间步是相同的，generator可以即时地从原始数据中生成样本，而不用保存每一个样本造成空间浪费。
>
> 因此，generator的元素不能直接索引，需要利用next()函数进行遍历。

```python
def generator(data, lookback, delay, min_index, max_index,
              shuffle=False, batch_size=128, step=6):
    if max_index is None:
        max_index = len(data) - delay - 1
    i = min_index + lookback
    while 1:
        if shuffle:
            rows = np.random.randint(
                min_index + lookback, max_index, size=batch_size)
        else:
            if i + batch_size >= max_index:
                i = min_index + lookback
            rows = np.arange(i, min(i + batch_size, max_index))
            i += len(rows)

        samples = np.zeros((len(rows),
                           lookback // step,
                           data.shape[-1]))
        targets = np.zeros((len(rows),))
        for j, row in enumerate(rows):
            indices = range(rows[j] - lookback, rows[j], step)
            samples[j] = data[indices]
            targets[j] = data[rows[j] + delay][1]
        yield samples, targets    #生成一个元组(samples,targets)，samples是输入数据批量，targets是对应的目标温度数组。
```

**samples结构**：（3D张量）每个时间步包含14个气象指标，为一个14D向量；每十天共lookback(1440/6)个时间步被编码成一个(1440/6,14)的2D张量；而每batch_size(128)个2D张量组成的样本组合又被编码为一个(128,1440/6,14)的3D张量。

**targets结构**：与samples相似，为领先了delay(1440/6)个时间步的(128,1440/6,1)的3D张量。

*准备训练生成器、验证生成器和测试生成器：*实例化三个生成器。训练生成器读取前200000个时间步，验证生成器读取随后的100000个时间步，测试生成器读取剩下的时间步。

```python
lookback = 1440
step = 6
delay = 144
batch_size = 128

train_gen = generator(float_data,    #训练集
                      lookback=lookback,
                      delay=delay,
                      min_index=0,
                      max_index=200000,
                      shuffle=True,
                      step=step,
                      batch_size=batch_size)
val_gen = generator(float_data,    #验证集
                    lookback=lookback,
                    delay=delay,
                    min_index=200001,
                    max_index=300000,
                    step=step,
                    batch_size=batch_size)
test_gen = generator(float_data,    #测试集
                     lookback=lookback,
                     delay=delay,
                     min_index=300001,
                     max_index=None,
                     step=step,
                     batch_size=batch_size)

# This is how many steps to draw from <code data-enlighter-language="raw" class="EnlighterJSRAW">val_gen</code>
# in order to see the whole validation set:
val_steps = (300000 - 200001 - lookback) // batch_size

# This is how many steps to draw from <code data-enlighter-language="raw" class="EnlighterJSRAW">test_gen</code>
# in order to see the whole test set:
test_steps = (len(float_data) - 300001 - lookback) // batch_size
```

训练集、验证集和测试集中都含有多个(sample, target)元组，个数可由var_steps, test_steps求得。

### 一种基于常识的预测方法

假设当日数据与前一日数据相等，计算其平均绝对误差。作为基准模型。

```python
def evaluate_naive_method():
    batch_maes = []
    for step in range(val_steps):
        samples, targets = next(val_gen)
        preds = samples[:, -1, 1]
        mae = np.mean(np.abs(preds - targets))
        batch_maes.append(mae)
    print(np.mean(batch_maes))

evaluate_naive_method()
```

输出为0.2897359729905486

### 一种基本的机器学习方法

在尝试代价较高的RNN之前，运行一个小型的密集连接网络作为基准。

_训练并评估一个密集连接模型：_

```python
from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop

model = Sequential()
model.add(layers.Flatten(input_shape=(lookback // step, float_data.shape[-1])))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(1))

model.compile(optimizer=RMSprop(), loss='mae')
history = model.fit_generator(train_gen,
                              steps_per_epoch=500,
                              epochs=20,
                              validation_data=val_gen,
                              validation_steps=val_steps)
```

_绘制结果：_

```python
import matplotlib.pyplot as plt

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(loss))

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')    #训练损失，优化程度
plt.plot(epochs, val_loss, 'b', label='Validation loss')    #验证损失，泛化程度
plt.title('Training and validation loss')
plt.legend()

plt.show()
```

![](@assets/images/21-3-300x213.png)

可以看出密集连接神经网络得到的结果并不好。

### 使用RNN

_训练并评估一个基于GRU的模型：_

```python
from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop

model = Sequential()
model.add(layers.GRU(32, input_shape=(None, float_data.shape[-1])))
model.add(layers.Dense(1))

model.compile(optimizer=RMSprop(), loss='mae')
history = model.fit_generator(train_gen,
                              steps_per_epoch=500,
                              epochs=20,
                              validation_data=val_gen,
                              validation_steps=val_steps)
```

_查看结果：_

```python
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(loss))

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()
```

![](@assets/images/21-4-300x209.png)

## 过拟合的判断和解决方案

### 过拟合的概念：

- 优化：调节模型以在训练数据上得到最佳性能。
- 泛化：训练好的模型在前所未见的数据上的性能好坏。

训练开始时，训练数据上的损失(training loss)越小，测试数据上的损失(validation loss)越小，这时模型就是欠拟合的。

随着神经网络在训练数据上迭代了一定次数后，泛化不再提高，验证指标先是不变，然后开始变差，即模型开始过拟合。

### 解决方案：

- 减小网络大小。
- 添加权重正则化。
- 添加dropout正则化。

## RNN的高级应用

### 使用循环dropout来降低过拟合

dropout是神经网络最有效也最常用的正则化方法之一。dropout比率是被设为0的特征所占的比例，通常在0.2~0.5范围内。

_训练并评估一个使用dropout正则化的基于GRU的模型：_

```python
from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop

model = Sequential()
model.add(layers.GRU(32,
                     dropout=0.2,
                     recurrent_dropout=0.2,
                     input_shape=(None, float_data.shape[-1])))
model.add(layers.Dense(1))

model.compile(optimizer=RMSprop(), loss='mae')
history = model.fit_generator(train_gen,
                              steps_per_epoch=500,
                              epochs=40,
                              validation_data=val_gen,
                              validation_steps=val_steps)
```

_绘制结果：_

```python
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(loss))

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()
```

### 循环层堆叠

模型如果不再过拟合，但遇到了性能瓶颈，就可以考虑增加网络容量。

在Keras中逐个堆叠循环层，所有中间层都应该返回完整的输出序列（一个3D张量），而不只是返回最后一个时间步的输出。可以通过指定return_sequences=True来实现。

_训练并评估一个使用dropout正则化的堆叠GRU模型：_

```python
from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop

model = Sequential()
model.add(layers.GRU(32,
                     dropout=0.1,
                     recurrent_dropout=0.5,
                     return_sequences=True,
                     input_shape=(None, float_data.shape[-1])))
model.add(layers.GRU(64, activation='relu',
                     dropout=0.1,
                     recurrent_dropout=0.5))
model.add(layers.Dense(1))

model.compile(optimizer=RMSprop(), loss='mae')
history = model.fit_generator(train_gen,
                              steps_per_epoch=500,
                              epochs=40,
                              validation_data=val_gen,
                              validation_steps=val_steps)
```

_绘制结果：_

```python
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(loss))

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()
```

## 利用模型进行预测

_对所有样本进行预测：_

```python
def predict_test():
    predict_result=[]
    for step in range(test_steps):
        samples, targets = next(test_gen)
        pr=model.predict(samples)
        predict_result.append(pr)
    print(predict_result)

predict_test()
```

输出结果的结构为每一个样本中的samples对应的结果array的组合，单个array长度为128(batch_size)。总array数目为test_steps。

\_对单个批量的样本进行预测：\_generator需要依序遍历

```python
def getElementbByIndex(index,gen):#取生成器中的第几个元素，从0开始
    r=None
    for i in range(index):
        try:
            r=next(gen)
        except :
            return None
    return r

def predict_test(index):
    samples, targets=getElementbByIndex(index,test_gen)
    predict_result=model.predict(samples)
    print(predict_result)

predict_test(300)    #这里index取300
```

输出结构为长度为128(batch_size)的数组，即对1个batch/128个samples预测的结果。

至此，预测貌似不能再细化了。predict()内的参数应该至少含有一个batch。

## Keras保存模型和加载模型

保存模型：

```python
from keras.models import load_model
model.save('my_model.h5')
```

加载模型：

```python
from keras.models import load_model
path='...'
my_model = load_model(path)
```

## **参考文献**

- **_《python深度学习》——[美] 弗朗索瓦·肖莱_**

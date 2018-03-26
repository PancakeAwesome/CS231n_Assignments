# -*- coding: UTF-8 -*-
import random
import numpy as np
from cs231n.data_utils import load_CIFAR10
# import matplotlib.pyplot as plt
# %matpblotlib inline
# plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
# plt.rcParams['image.interpolation'] = 'nearest'
# plt.rcParams['image.cmap'] = 'gray'

# for auto-reloading extenrnal modules
# see http://stackoverflow.com/questions/1907993/autoreload-of-modules-in-ipython
# %load_ext autoreload
# %autoreload 2

from cs231n.classifiers.neural_net import TwoLayerNet

def rel_error(x, y):
  """ returns relative error """
  return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))

# Create a small net and some toy data to check your implementations.
# Note that we set the random seed for repeatable experiments.

input_size = 4
hidden_size = 10
num_classes = 3
num_inputs = 5

def init_toy_model():
  np.random.seed(0)
  return TwoLayerNet(input_size, hidden_size, num_classes, std=1e-1)

def init_toy_data():
  np.random.seed(1)
  X = 10 * np.random.randn(num_inputs, input_size)
  y = np.array([0, 1, 2, 2, 1])
  return X, y

net = init_toy_model()
X, y = init_toy_data()

# 前向传播：计算分值
scores = net.loss(X)
print 'Your scores:'
print scores
print 'correct scores:'
correct_scores = np.asarray([
  [-0.81233741, -1.27654624, -0.70335995],
  [-0.17129677, -1.18803311, -0.47310444],
  [-0.51590475, -1.01354314, -0.8504215 ],
  [-0.15419291, -0.48629638, -0.52901952],
  [-0.00618733, -0.12435261, -0.15226949]])
print correct_scores

# The difference should be very small. We get < 1e-7
print 'Difference between your scores and correct scores:'
print np.sum(np.abs(scores - correct_scores))

# 观察损失值
loss, _ = net.loss(X, y)
correct_loss = 1.30378789133

# should be very small, we get < 1e-12
print 'Difference between your loss and correct loss:'
print np.sum(np.abs(loss - correct_loss))

# 反向传播
from cs231n.gradient_check import eval_numerical_gradient

# 使用数值梯度检查法检查反向传播的实现
loss, grads = net.loss(X, y, reg = 0.05)

# 这些都应该小于1e-8
for param_name in grads:
  f = lambda W: net.loss(X, y, reg=0.1)[0]
  param_grad_num = eval_numerical_gradient(f, net.params[param_name], verbose=False)
  print '%s max relative error: %e' % (param_name, rel_error(param_grad_num, grads[param_name]))

# 训练神经网络
net = init_toy_model()
stats = net.train(X, y, X, y, learning_rate = 1e-1, reg = 5e-6, num_iters = 100, verbose = False)

# 观察最后一个损失值
print('Final training loss:', stats['loss_history'][-1])

# plot the loss history
# plt.plot(stats['loss_history'])
# plt.xlabel('iteration')
# plt.ylabel('training loss')
# plt.title('Training Loss history')
# plt.show()

# 加载cifar10图像数据
from cs231n.data_utils import load_CIFAR10

def get_CIFAR10_data(num_training = 49000, num_validation = 1000, num_test = 1000):
    # 加载cifar-10数据
    cifar10_dir = 'cs231n/datasets/cifar-10-batches-py'
    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

    #从数据集中取子数据集用于后面的练习
    mask = list(range(num_training, num_training + num_validation))
    X_val = X_train[mask]
    y_val = y_train[mask]
    mask = list(range(num_training))
    X_train = X_train[mask]
    y_train = y_train[mask]
    mask = list(range(num_test))
    X_test = X_train[mask]
    y_test = y_train[mask]

    #标准化数据：先求平均图像，再将每个图像都减去平均图像
    mean_image = np.mean(X_train, axis = 0)
    X_train -= mean_image
    X_val -= mean_image
    X_test -= mean_image

    # 将图像数据变成2维
    X_train = X_train.reshape(num_training, -1)
    X_val = X_val.reshape(num_validation, -1)
    X_test = X_test.reshape(num_test, -1)

    return X_train, y_train, X_val, y_val, X_test, y_test

# 调用该方法获取我们的数据
# Invoke the above function to get our data.
X_train, y_train, X_val, y_val, X_test, y_test = get_CIFAR10_data()
print 'Train data shape: ', X_train.shape
print 'Train labels shape: ', y_train.shape
print 'Validation data shape: ', X_val.shape
print 'Validation labels shape: ', y_val.shape
print 'Test data shape: ', X_test.shape
print 'Test labels shape: ', y_test.shape    

# 训练神经网络
input_size = 32 * 32 * 3
hidden_size = 50
num_classes = 10
net = TwoLayerNet(input_size, hidden_size, num_classes)

# 开始训练
stats = net.train(X_train, y_train, X_val, y_val, num_iters = 1000, batch_size = 200, learning_rate = 1e-4, learning_rate_decay = 0.95, reg = 0.25, verbose = True)

# 在验证集上进行预测
val_acc = (net.predict(X_val) == y_val).mean()
print('Validation accuracy:', val_acc)

# 调试训练结果
# 1:可以绘制损失函数值以及在优化过程中训练集和验证集之间的准确性
# Plot the loss function and train / validation accuracies
# plt.subplot(2, 1, 1)
# plt.plot(stats['loss_history'])
# plt.title('Loss history')
# plt.xlabel('Iteration')
# plt.ylabel('Loss')

# plt.subplot(2, 1, 2)
# plt.plot(stats['train_acc_history'], label='train')
# plt.plot(stats['val_acc_history'], label='val')
# plt.title('Classification accuracy history')
# plt.xlabel('Epoch')
# plt.ylabel('Clasification accuracy')
# plt.show()

# 2:可视化第一层神经网络学习到的权重
from cs231n.vis_utils import visualize_grid

# 可视化网络的权重
# def show_net_weights(net):
#     W1 = net.params['W1']
#     W1 = W1.reshape(32, 32, 3, -1).transpose(3, 0, 1, 2)# 转成初试图像的维数
#     plt.imshow(visualize_grid(W1, padding=3).astype('uint8'))
#     plt.gca().axis('off')
#     plt.show()

# show_net_weights(net)

# 调试超参数
# 1:通过交叉验证选择最合适的超参数组合
best_net = None
# 将最好的模型放到里面

best_val = -1
best_stats = None
learning_rates = [1e-2, 1e-3]
regularization_strengths = [0.4, 0.5, 0.6]
results = {}
iters = 2000 # 100
for lr in learning_rates:
    for rs in regularization_strengths:
        net = TwoLayerNet(input_size, hidden_size, num_classes)

        # 训练网络
        stats = net.train(X_train, y_train, X_val, y_val, num_iters = iters, batch_size = 200, learning_rate = lr, learning_rate_decay = 0.95, reg = rs)

        y_train_pred = net.predict(X_train)
        acc_train = np.mean(y_train_pred == y_train)
        y_val_pred = net.predict(X_val)
        acc_val = np.mean(y_val_pred == y_val)

        results[(lr, rs)] = (acc_train, acc_val)

        if best_val < acc_val:
            best_val = acc_val
            best_stats = stats
            best_net = net

# 打印出结果
for lr, reg in sorted(results):
    train_accuracy, val_accuracy = results[(lr, reg)]
    print 'lr %e reg %e train accuracy: %f val accuracy: %f' % (lr, reg, train_accuracy, val_accuracy)
    
print 'best validation accuracy achieved during cross-validation: %f' % best_val

# 可视化结果
# show_net_weights(best_net)

# 在测试集上测试结果
test_acc = (best_net.predict(X_test) == y_test).mean()
print 'Test accuracy: ', test_acc

# 使用两个特征提取的方法
# 1: HOG（梯度直方图）：捕捉图像的纹理特征
# 2: HSV（色调，饱和度，明度）：捕捉图像的颜色特征
from cs231n.features import *
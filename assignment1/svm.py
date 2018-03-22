# -*- coding: UTF-8 -*-
# Run some setup code for this notebook.
import random
import numpy as np
from cs231n.data_utils import load_CIFAR10
# import matplotlib.pyplot as plt

# from __future__ import print_function
# from past.builtins import xrange

# This is a bit of magic to make matplotlib figures appear inline in the
# notebook rather than in a new window.
# %matplotlib inline
# plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
# plt.rcParams['image.interpolation'] = 'nearest'
# plt.rcParams['image.cmap'] = 'gray'

# Some more magic so that the notebook will reload external python modules;
# see http://stackoverflow.com/questions/1907993/autoreload-of-modules-in-ipython
# %load_ext autoreload
# %autoreload 2

#加载cifar-10原始数据
# Load the raw CIFAR-10 data.
cifar10_dir = 'cs231n/datasets/cifar-10-batches-py'
X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

# As a sanity check, we print out the size of the training and test data.
print ('Training data shape: ', X_train.shape)
print 'Training labels shape: ', y_train.shape
print 'Test data shape: ', X_test.shape
print 'Test labels shape: ', y_test.shape

# 挑选几个样本查看
# Visualize some examples from the dataset.
# We show a few examples of training images from each class.
classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
num_classes = len(classes)
samples_per_class = 7
# for y, cls in enumerate(classes):
#     idxs = np.flatnonzero(y_train == y)
#     idxs = np.random.choice(idxs, samples_per_class, replace=False)
#     for i, idx in enumerate(idxs):
#         plt_idx = i * num_classes + y + 1
#         plt.subplot(samples_per_class, num_classes, plt_idx)
#         plt.imshow(X_train[idx].astype('uint8'))
#         plt.axis('off')
#         if i == 0:
#             plt.title(cls)
# plt.show()

# 数据分割成训练集，验证集和测试集
# dev开发集 等同于验证集
num_training = 49000
num_validation = 1000
num_test = 1000
num_dev = 500

# 验证集将是从原始训练集分割出来的长度为num_validation的数据样本点
mask = range(num_training, num_training + num_validation)
X_val = X_train[mask]
y_val = y_train[mask]

# 训练集是原始的训练集中前num_train个样本
mask = range(num_training)
X_train = X_train[mask]
y_train = y_train[mask]

# 可以从训练集中随机抽取一小部分的数据点作为开发集
mask = np.random.choice(num_training, num_dev, replace = False)
X_dev = X_train[mask]
y_dev = y_train[mask]

# 使用前num_test个测试集点作为测试集
mask = range(num_test)
X_test = X_train[mask]
y_test = y_train[mask]

print 'Train data shape: ', X_train.shape
print 'Train labels shape: ', y_train.shape
print 'Validation data shape: ', X_val.shape
print 'Validation labels shape: ', y_val.shape
print 'Test data shape: ', X_test.shape
print 'Test labels shape: ', y_test.shape

# 数据预处理，将原始数据转成二维数据
X_train = np.reshape(X_train, (X_train.shape[0], -1))
X_val = np.reshape(X_val, (X_val.shape[0], -1))
X_dev = np.reshape(X_dev, (X_dev.shape[0], -1))
X_test = np.reshape(X_test, (X_test.shape[0], -1))

# 打印一下
# As a sanity check, print out the shapes of the data
print 'Training data shape: ', X_train.shape
print 'Validation data shape: ', X_val.shape
print 'Test data shape: ', X_test.shape
print 'dev data shape: ', X_dev.shape

# 预处理， 均值零中心化
# 首先基于训练数据，计算图像的平均值
mean_image = np.mean(X_train, axis = 0)# 计算出每维特征的平均值
print mean_image.shape
print mean_image[:10]
# plt.figure(figsize = (4,4))
# 均值可视化
# plt.imshow(mean_image.reshape((32,32,3)).astype('uint8')) # visualize the mean image
# plt.show()

# 将训练数据和测试集图像分别减去平均值
X_train -= mean_image
X_test -= mean_image
X_val -= mean_image
X_dev -= mean_image

# 将偏置和权重合并
X_train = np.hstack((X_train, np.ones((X_train.shape[0], 1))))
X_test = np.hstack((X_test, np.ones((X_test.shape[0], 1))))
X_val = np.hstack((X_val, np.ones((X_val.shape[0], 1))))
X_dev = np.hstack((X_dev, np.ones((X_dev.shape[0], 1))))

print(X_train.shape, X_test.shape, X_val.shape, X_test.shape)

# 评估loss(朴素)
from cs231n.classifiers.linear_svm import svm_loss_naive
import time

# 生成一个很小的随机权重矩阵
# 标准正态随机然后乘0.0001
W = np.random.randn(3073, 10) * 0.0001

loss, grad = svm_loss_naive(W, X_dev, y_dev, 0.000005)
print('loss: %f' % (loss,))

# 实现梯度之后，运行下面的代码重新计算梯度
loss, grad = svm_loss_naive(W, X_dev, y_dev, 0.0)

# 执行梯度检查
from cs231n.gradient_check import grad_check_sparse

# 匿名函数
f = lambda w: svm_loss_naive(w,X_dev, y_dev, 0)[0]
grad_numerical = grad_check_sparse(f, W, grad)

# 再次验证梯度 这次使用正则项 
print('turn on reg')
loss, grad = svm_loss_naive(W, X_dev, y_dev, 5e1)
f = lambda w: svm_loss_naive(w,X_dev, y_dev, 5e1)[0]
grad_numerical = grad_check_sparse(f, W, grad)

# Next implement the function svm_loss_vectorized; for now only compute the loss;
# we will implement the gradient in a moment.
tic = time.time()
loss_naive, grad_naive = svm_loss_naive(W, X_dev, y_dev, 0.00001)
toc = time.time()
print 'Naive loss: %e computed in %fs' % (loss_naive, toc - tic)

from cs231n.classifiers.linear_svm import svm_loss_vectorized
tic = time.time()
loss_vectorized, _ = svm_loss_vectorized(W, X_dev, y_dev, 0.00001)
toc = time.time()
print 'Vectorized loss: %e computed in %fs' % (loss_vectorized, toc - tic)

# The losses should match but your vectorized implementation should be much faster.
print 'difference: %f' % (loss_naive - loss_vectorized)

# 用朴素方法计算梯度和向量法进行比较，向量法意会更快一点
# Complete the implementation of svm_loss_vectorized, and compute the gradient
# of the loss function in a vectorized way.

# The naive implementation and the vectorized implementation should match, but
# the vectorized version should still be much faster.
tic = time.time()
_, grad_naive = svm_loss_naive(W, X_dev, y_dev, 0.00001)
toc = time.time()
print 'Naive loss and gradient: computed in %fs' % (toc - tic)

tic = time.time()
_, grad_vectorized = svm_loss_vectorized(W, X_dev, y_dev, 0.00001)
toc = time.time()
print 'Vectorized loss and gradient: computed in %fs' % (toc - tic)

# The loss is a single number, so it is easy to compare the values computed
# by the two implementations. The gradient on the other hand is a matrix, so
# we use the Frobenius norm to compare them.
difference = np.linalg.norm(grad_naive - grad_vectorized, ord='fro')
print 'difference: %f' % difference

# 查看梯度下降法的效果
# In the file linear_classifier.py, implement SGD in the function
# LinearClassifier.train() and then run it with the code below.
from cs231n.classifiers import LinearSVM
svm = LinearSVM()
tic = time.time()
loss_hist = svm.train(X_train, y_train, learning_rate=1e-7, reg=5e4,
                      num_iters=1500, verbose=True)
toc = time.time()
print 'That took %fs' % (toc - tic)

# 将损失曲线画出来
# A useful debugging strategy is to plot the loss as a function of
# iteration number:
# plt.plot(loss_hist)
# plt.xlabel('Iteration number')
# plt.ylabel('Loss value')
# plt.show()

# 在训练集和验证集上分别预测一下
y_train_pred = svm.predict(X_train)
print 'training accuracy: %f' % (np.mean(y_train == y_train_pred), )
y_val_pred = svm.predict(X_val)
print 'validation accuracy: %f' % (np.mean(y_val == y_val_pred), )

# 使用验证集调整超参数（正则化强度和学习率）

# 尝试不同的学习率和正则化强度
# 先尝试用较大的步长搜索，再微调

learning_rates = [2e-7, 0.75e-7, 1.5e-7, 1.25-7, 0.75e-7]
regularization_strengths = [3e4, 3.25e4, 3.5e4, 3.75e4, 4e4, 4.25e4, 4.5e4, 4.75e4, 5e4]

# 结果是一个词典dict，（learning_rate, regularization）   

results = {}
best_val = -1# 出现的最大准确率
best_svm = None# 出现的正确率最高的svm对象

################################################################################
# TODO:                                                                        #
# Write code that chooses the best hyperparameters by tuning on the validation #
# set. For each combination of hyperparameters, train a linear SVM on the      #
# training set, compute its accuracy on the training and validation sets, and  #
# store these numbers in the results dictionary. In addition, store the best   #
# validation accuracy in best_val and the LinearSVM object that achieves this  #
# accuracy in best_svm.                                                        #
#                                                                              #
# Hint: You should use a small value for num_iters as you develop your         #
# validation code so that the SVMs don't take much time to train; once you are #
# confident that your validation code works, you should rerun the validation   #
# code with a larger value for num_iters.                                      #
################################################################################
#先使用较小的num_iters这样训练的时间不会很长，当确认code可以正常运行之后，再用较大的num_iters重新跑验证代码

for rate in learning_rates:
    for regular in regularization_strengths:
        svm = LinearSVM()
        svm.train(X_train, y_train, learning_rate = rate, reg = regular, num_iters = 1000)
        y_train_pred = svm.predict(X_train)
        accuracy_train = np.mean(y_train == y_train_pred)
        y_val_pred = svm.predict(X_val)
        accuracy_val = np.mean(y_val == y_val_pred)
        results[(rate, regular)] = (accuracy_train, accuracy_val)
        if best_val < accuracy_val:
        	best_val = accuracy_val
        	best_svm = svm
for lr, reg in results:
	train_accuracy, val_accuracy = results[(lr, reg)]	
	print 'lr %e reg %e train accuracy: %f val accuracy: %f' % (
                lr, reg, train_accuracy, val_accuracy)
    
print 'best validation accuracy achieved during cross-validation: %f' % best_val

# 可视化交叉验证结果
# Visualize the cross-validation results
# import math
# x_scatter = [math.log10(x[0]) for x in results]
# y_scatter = [math.log10(x[1]) for x in results]

# # plot training accuracy
# marker_size = 100
# colors = [results[x][0] for x in results]
# plt.subplot(2, 1, 1)
# plt.scatter(x_scatter, y_scatter, marker_size, c=colors)
# plt.colorbar()
# plt.xlabel('log learning rate')
# plt.ylabel('log regularization strength')
# plt.title('CIFAR-10 training accuracy')

# # plot validation accuracy
# colors = [results[x][1] for x in results] # default size of markers is 20
# plt.subplot(2, 1, 2)
# plt.scatter(x_scatter, y_scatter, marker_size, c=colors)
# plt.colorbar()
# plt.xlabel('log learning rate')
# plt.ylabel('log regularization strength')
# plt.title('CIFAR-10 validation accuracy')
# plt.show()

# 在测试集上评价最好的svm的表现
y_test_pred = best_svm.predict(X_test)
test_accuracy = np.mean(y_test == y_test_pred)
print 'linear SVM on raw pixels final test set accuracy: %f' % test_accuracy

# Visualize the learned weights for each class.
# Depending on your choice of learning rate and regularization strength, these may
# or may not be nice to look at.
w = best_svm.W[:-1,:] # strip out the bias
w = w.reshape(32, 32, 3, 10)
w_min, w_max = np.min(w), np.max(w)
classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
# for i in xrange(10):
#   plt.subplot(2, 5, i + 1)
    
#   # Rescale the weights to be between 0 and 255
#   wimg = 255.0 * (w[:, :, :, i].squeeze() - w_min) / (w_max - w_min)
#   plt.imshow(wimg.astype('uint8'))
#   plt.axis('off')
#   plt.title(classes[i])


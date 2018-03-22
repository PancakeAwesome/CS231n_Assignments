# -*- coding: UTF-8 -*-
# Run some setup code for this notebook.
# from __future__ import print_function
# from past.builtins import xrange

import random
import numpy as np
from cs231n.data_utils import load_CIFAR10
import matplotlib.pyplot as plt

# This is a bit of magic to make matplotlib figures appear inline in the notebook
# rather than in a new window.
# %matplotlib inline
plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# Some more magic so that the notebook will reload external python modules;
# see http://stackoverflow.com/questions/1907993/autoreload-of-modules-in-ipython
# %load_ext autoreload
# %autoreload 2
# 
# 加载CIFAR-10数据
cifar10_dir = 'cs231n/datasets/cifar-10-batches-py'
X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

# 打印训练集和测试集的数据
print ('Training data shape: ', X_train.shape)
print ('Training labels shape: ', y_train.shape)
print ('Test data shape: ', X_test.shape)
print ('Test labels shape: ', y_test.shape)

# 查看数据集中的样本
# 每一类随机挑出几个样本
classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
num_classes = len(classes)
samples_per_class = 7
for y, cls in enumerate(classes):
    idxs = np.flatnonzero(y_train == y)
    idxs = np.random.choice(idxs, samples_per_class, replace = False)
    for i, idx in enumerate(idxs):
        plt_idx = i * num_classes + y + 1
        plt.subplot(samples_per_class, num_classes, plt_idx)
        plt.imshow(X_train[idx].astype('uint8'))
        plt.axis('off')
        if i == 0:
            plt.title(cls)
plt.show()

# 为了更高效的执行代码，取数据集的子集来训练和测试
num_training = 5000
mask = range(num_training)
X_train = X_train[mask]
y_train = y_train[mask]

num_test = 500
mask = range(num_test)
X_test = X_test[mask]
y_test = y_test[mask]

# 将数据转成二维的
X_train = np.reshape(X_train, (X_train.shape[0], -1))
X_test = np.reshape(X_test, (X_test.shape[0], -1))
print (X_train.shape, X_test.shape)

# 创建knn分类器对象
# 注意：knn分类器不进行操作，只是将训练数据进行了简单的存储
from cs231n.classifiers import KNearestNeighbor
classifier = KNearestNeighbor()
classifier.train(X_train, y_train)

# 测试一下
dists = classifier.compute_distances_two_loops(X_test)
print(dists.shape)

# 我们可以将距离矩阵进行可视化：其中每一行表示一个测试样本与所有训练样本的距离
plt.imshow(dists, interpolation='none')
plt.show()

# 将k设置为1（也就是最近邻算法）
y_test_pred = classifier.predict_labels(dists, k = 1)

#计算并打印准确率
num_correct = np.sum(y_test_pred == y_test)
accuracy = float(num_correct) / num_test
print 'Got %d / %d correct => accuracy: %f' % (num_correct, num_test, accuracy)

# 提高k的大小，令k=5
y_test_pred = classifier.predict_labels(dists, k = 5)

#计算并打印准确率
num_correct = np.sum(y_test_pred == y_test)
accuracy = float(num_correct) / num_test
print 'Got %d / %d correct => accuracy: %f' % (num_correct, num_test, accuracy)

dists_one = classifier.compute_distances_one_loop(X_test)

# 为保证向量化的代码运行正确，将运行结果与前面的结果进行对比。对比两个矩阵的方法有很多，比较简单的是使用Frobenius范数，其表示两个矩阵所有元素的差值的均方根。或者可以将两个矩阵reshanpe成向量后，比较它们之间的欧式距离
difference = np.linalg.norm(dists - dists_one, ord = 'fro')
print 'Difference was: %f' % (difference, )
if difference < 0.001:
    print 'Good! The distance matrices are the same'
else:
    print 'Uh-oh! The distance matrices are different'

# 全向量计算距离
dists_two = classifier.compute_distances_no_loops(X_test)

# 和之前的结果相比较
difference = np.linalg.norm(dists - dists_two, ord = 'fro')
print('Difference was: %f' % (difference,))
if difference < 0.001:
    print('God! The distance matrices are the same!')
else:
    print('Uh-oh! The distance matrices are different')

# 对比一下三种计算距离的方式的执行速度
# Let's compare how fast the implementations are
def time_function(f, *args):
  """
  Call a function f with args and return the time (in seconds) that it took to execute.
  """
  import time
  tic = time.time()
  f(*args)
  toc = time.time()
  return toc - tic

two_loop_time = time_function(classifier.compute_distances_two_loops, X_test)
print 'Two loop version took %f seconds' % two_loop_time

one_loop_time = time_function(classifier.compute_distances_one_loop, X_test)
print 'One loop version took %f seconds' % one_loop_time

no_loop_time = time_function(classifier.compute_distances_no_loops, X_test)
print 'No loop version took %f seconds' % no_loop_time


# 通过交叉验证来寻找最优超参数k
num_folds = 5
k_choices = [1, 3, 5, 8, 10, 12, 15, 20, 50, 100]

X_train_folds = []
y_train_folds = []

################################################################################
# TODO:                                                                        #
# Perform k-fold cross validation to find the best value of k. For each        #
# possible value of k, run the k-nearest-neighbor algorithm num_folds times,   #
# where in each case you use all but one of the folds as training data and the #
# last fold as a validation set. Store the accuracies for all fold and all     #
# values of k in the k_to_accuracies dictionary.                               #
################################################################################
# 将训练数据切分成不同的折。切分之后，训练样本和对应的样本标签被包含在数组X_train_folds和y_train_folds之中
# y_train_folds[i]是一个矢量，表示X_train_folds[i]中所有样本的标签
# 可以使用numpy的array_split方法
X_train_folds = np.array_split(X_train, num_folds)
y_train_folds = np.array_split(y_train, num_folds)
################################################################################
#                                 END OF YOUR CODE                             #
################################################################################

# 将不同k值下的准确率保存在一个字典中。
k_to_accuracies = {}

# 交叉验证：执行knn算法num_folds次，每次选择一折为验证集，其他折为训练集，将准确率保存在k_to_accuracy中。
classifier = KNearestNeighbor()
for k in k_choices:
    accuracies = np.zeros(num_folds)
    for fold in xrange(num_folds):
        temp_X = X_train_folds[:]
        temp_y = y_train_folds[:]
        X_validate_fold = temp_X.pop(fold)
        y_validate_fold = temp_y.pop(fold)

        temp_X = np.array([y for x in temp_X for y in x])
        temp_y = np.array([y for x in temp_y for y in x])
        classifier.train(temp_X, temp_y)

        y_test_pred = classifier.predict(X_validate_fold, k = k)
        num_correct = np.sum(y_test_pred == y_validate_fold)
        accuracy = float(num_correct) / len(y_test_pred)
        accuracies[fold] = accuracy
    k_to_accuracies[k] = accuracies    

# 输出准确率
for k in sorted(k_to_accuracies):
    for accuracy in k_to_accuracies[k]:
        print 'k = %d, accuracy = %f' % (k, accuracy)

# 画图
for k in k_choices:
  accuracies = k_to_accuracies[k]
  plt.scatter([k] * len(accuracies), accuracies)

# plot the trend line with error bars that correspond to standard deviation
accuracies_mean = np.array([np.mean(v) for k,v in sorted(k_to_accuracies.items())])
accuracies_std = np.array([np.std(v) for k,v in sorted(k_to_accuracies.items())])
plt.errorbar(k_choices, accuracies_mean, yerr=accuracies_std)
plt.title('Cross-validation on k')
plt.xlabel('k')
plt.ylabel('Cross-validation accuracy')
plt.show()

#根据交叉验证的结果，选择最优的k，然后在全量数据上进行实验，你将得到超过28%的准确率
best_k = 10

classifier = KNearestNeighbor()
classifier.train(X_train, y_train)
y_test_pred = classifier.predict(X_test, k = best_k)

# 计算并显示准确率  
num_correct = np.sum(y_test_pred == y_train)
accuracy = float(num_correct) / num_test
print('Got %d / %d correct => accuracy: %f' % (num_correct, num_test, accuracy))
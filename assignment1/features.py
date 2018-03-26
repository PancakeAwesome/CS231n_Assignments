# -*- coding: UTF-8 -*-
import random
import numpy as np
from cs231n.data_utils import load_CIFAR10
import matplotlib.pyplot as plt
# %matplotlib inline
plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# for auto-reloading extenrnal modules
# see http://stackoverflow.com/questions/1907993/autoreload-of-modules-in-ipython
# %load_ext autoreload
# %autoreload 2

from cs231n.features import color_histogram_hsv, hog_feature

def get_CIFAR10_data(num_training=49000, num_validation=1000, num_test=1000):
  # Load the raw CIFAR-10 data
  cifar10_dir = 'cs231n/datasets/cifar-10-batches-py'
  X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)
  
  # Subsample the data
  mask = range(num_training, num_training + num_validation)
  X_val = X_train[mask]
  y_val = y_train[mask]
  mask = range(num_training)
  X_train = X_train[mask]
  y_train = y_train[mask]
  mask = range(num_test)
  X_test = X_test[mask]
  y_test = y_test[mask]

  return X_train, y_train, X_val, y_val, X_test, y_test

X_train, y_train, X_val, y_val, X_test, y_test = get_CIFAR10_data()

# 特征提取
from cs231n.features import *

num_color_bins = 10 # hog特征提取中的直方图特征数目
# 两个特征提取函数
# 1: hog_feature:用来提取图片的纹理特征，改写了scikit-image的fog接口，并且要首先转换成灰度图。
# 2: color_histogram_hsv: 用来提取图片的颜色特征HSV，颜色直方图是实现用matplotlib.colors.rgb_to_hsv
# 的接口把图片从RGB变成HSV，再提取明度，把value投射到不同的bin中去
# 分别用这两个提取特征的函数对每个数据集进行特征提取
features_fns = [hog_feature, lambda img: color_histogram_hsv(img, nbin = num_color_bins)]
X_train_feats = extract_features(X_train, features_fns, verbose = True)
X_val_feats = extract_features(X_val, features_fns)
X_test_feats = extract_features(X_test, features_fns)

print(X_train_feats.shape, X_val_feats.shape, X_test_feats.shape)

# 预处理：减去每一列特征的平均值
mean_feat = np.mean(X_train_feats, axis = 0, keepdims = True)
X_train_feats -= mean_feat
X_val_feats -= mean_feat
X_test_feats -= mean_feat

# 预处理：每一列除以标准差，这确保了每个特征都在一个数值范围
std_feat = np.std(X_train_feats, axis = 0, keepdims = True)
X_train_feats /= std_feat
X_val_feats /= std_feat
X_test_feats /= std_feat

# 多加一个bias列
X_train_feats = np.hstack((X_train_feats, np.ones((X_train_feats.shape[0], 1))))
X_val_feats = np.hstack((X_val_feats, np.ones((X_val_feats.shape[0], 1))))
X_test_feats = np.hstack((X_test_feats, np.ones((X_test_feats.shape[0], 1))))


# 用提取好特征的数据去训练SVM模型
from cs231n.classifiers.linear_classifier import LinearSVM

# learning_rates = [1e-9, 1e-8, 1e-7]
# regularization_strengths = [5e4, 5e5, 5e6]

# 用交叉验证调优超参数
results = {}
best_val = -1
best_svm = None

#pass
learning_rates = [5e-9, 7,5e-9, 1e-8]
regularization_strengths = [(5+i)*1e6 for i in range(-3, 4)]

################################################################################
# TODO:                                                                        #
# Use the validation set to set the learning rate and regularization strength. #
# This should be identical to the validation that you did for the SVM; save    #
# the best trained classifer in best_svm. You might also want to play          #
# with different numbers of bins in the color histogram. If you are careful    #
# you should be able to get accuracy of near 0.44 on the validation set.       #
################################################################################
for rs in regularization_strengths:
    for lr in learning_rates:
        svm = LinearSVM()
        loss_hist = svm.train(X_train_feats, y_train, lr, rs, num_iters=6000)
        y_train_pred = svm.predict(X_train_feats)
        train_accuracy = np.mean(y_train == y_train_pred)
        y_val_pred = svm.predict(X_val_feats)
        val_accuracy = np.mean(y_val == y_val_pred)
        if val_accuracy > best_val:
            best_val = val_accuracy
            best_svm = svm           
        results[(lr,rs)] = train_accuracy, val_accuracy
#pass
################################################################################
#                              END OF YOUR CODE                                #
################################################################################

# 打印出结果
# Print out results.
for lr, reg in sorted(results):
    train_accuracy, val_accuracy = results[(lr, reg)]
    print 'lr %e reg %e train accuracy: %f val accuracy: %f' % (
                lr, reg, train_accuracy, val_accuracy)
    
print 'best validation accuracy achieved during cross-validation: %f' % best_val

# 评价模型结果
# Evaluate your trained SVM on the test set
y_test_pred = best_svm.predict(X_test_feats)
test_accuracy = np.mean(y_test == y_test_pred)
print test_accuracy

# 想知道算法是如何运作的很重要的方法是，把它的分类错误可视化。
# An important way to gain intuition about how an algorithm works is to
# visualize the mistakes that it makes. In this visualization, we show examples
# of images that are misclassified by our current system. The first column
# shows images that our system labeled as "plane" but whose true label is
# something other than "plane".

examples_per_class = 8
classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
for cls, cls_name in enumerate(classes):
    idxs = np.where((y_test != cls) & (y_test_pred == cls))[0]
    idxs = np.random.choice(idxs, examples_per_class, replace=False)
    for i, idx in enumerate(idxs):
        plt.subplot(examples_per_class, len(classes), i * len(classes) + cls + 1)
        plt.imshow(X_test[idx].astype('uint8'))
        plt.axis('off')
        if i == 0:
            plt.title(cls_name)
plt.show()

# 用提取好特征的数据去训练神经网络模型
from cs231n.classifiers.neural_net import TwoLayerNet

input_dim = X_train_feats.shape[1]
hidden_dim = 500
num_classes = 10

net = TwoLayerNet(input_dim, hidden_dim, num_classes)

# 交叉验证调优模型（超参数）
################################################################################
# TODO: Train a two-layer neural network on image features. You may want to    #
# cross-validate various parameters as in previous sections. Store your best   #
# model in the best_net variable.                                              #
################################################################################

results = {}
best_val = -1
best_net = None

learning_rates = [1e-2 ,1e-1, 5e-1, 1, 5]
regularization_strengths = [1e-3, 5e-3, 1e-2, 1e-1, 0.5, 1]

for lr in learning_rates:
    for reg in regularization_strengths:
        net = TwoLayerNet(input_dim, hidden_dim, num_classes)
        # Train the network
        stats = net.train(X_train_feats, y_train, X_val_feats, y_val,
        num_iters=1500, batch_size=200,
        learning_rate=lr, learning_rate_decay=0.95,
        reg= reg, verbose=False)
        val_acc = (net.predict(X_val_feats) == y_val).mean()
        if val_acc > best_val:
            best_val = val_acc
            best_net = net         
        results[(lr,reg)] = val_acc

# Print out results.
for lr, reg in sorted(results):
    val_acc = results[(lr, reg)]
    print 'lr %e reg %e val accuracy: %f' % (
                lr, reg,  val_acc)
    
print 'best validation accuracy achieved during cross-validation: %f' % best_val
#pass
################################################################################
#                              END OF YOUR CODE                                #
################################################################################

# 测试集测试结果
# Run your neural net classifier on the test set. You should be able to
# get more than 55% accuracy.

test_acc = (best_net.predict(X_test_feats) == y_test).mean()
print test_acc

# 得到0.5的分类效果比不用图片特征提取（0.4）效果要好
# 到目前为止我们尝试了HOG和颜色直方图，但是其他类型的特征也许能得到更好的分类效果。
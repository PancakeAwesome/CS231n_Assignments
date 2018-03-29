# -*- coding: UTF-8 -*-
import numpy as np


def affine_forward(x, w, b):
  """
  Computes the forward pass for an affine (fully-connected) layer.

  The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
  examples, where each example x[i] has shape (d_1, ..., d_k). We will
  reshape each input into a vector of dimension D = d_1 * ... * d_k, and
  then transform it to an output vector of dimension M.

  Inputs:
  - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
  - w: A numpy array of weights, of shape (D, M)
  - b: A numpy array of biases, of shape (M,)
  
  Returns a tuple of:
  - out: output, of shape (N, M)
  - cache: (x, w, b)
  """
  out = None
  #############################################################################
  # TODO: Implement the affine forward pass. Store the result in out. You     #
  # will need to reshape the input into rows.                                 #
  #############################################################################
  reshape_x = np.reshape(x, (x.shape[0], -1))# 确保x是一个规整的矩阵
  out = np.dot(reshape_x, w) + b # out = w x + b
  #pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x, w, b) # 将函数的输入值缓存起来，以备后面计算梯度时使用
  return out, cache


def affine_backward(dout, cache):
  """
  Computes the backward pass for an affine layer.

  Inputs:
  - dout: Upstream derivative, of shape (N, M)
  - cache: Tuple of:
    - x: Input data, of shape (N, d_1, ... d_k)
    - w: Weights, of shape (D, M)

  Returns a tuple of:
  - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
  - dw: Gradient with respect to w, of shape (D, M)
  - db: Gradient with respect to b, of shape (M,)
  """
  x, w, b = cache # 从缓冲中读出x,w,b
  dx, dw, db = None, None, None
  #############################################################################
  # TODO: Implement the affine backward pass.                                 #
  #############################################################################
  reshape_x = np.reshape(x, (x.shape[0], -1)) # x: Input data, of shape (N, d_1, ... d_k)
  dx = np.reshape(np.dot(dout, w.T), x.shape)
  dw = np.dot(reshape_x.T, dout)
  db = np.sum(dout, axis = 0)
  #pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx, dw, db


def relu_forward(x):
  """
  Computes the forward pass for a layer of rectified linear units (ReLUs).

  Input:
  - x: Inputs, of any shape

  Returns a tuple of:
  - out: Output, of the same shape as x
  - cache: x
  """
  out = None
  #############################################################################
  # TODO: Implement the ReLU forward pass.                                    #
  #############################################################################
  out = np.maximum(0, x) # 取x中每个元素和0作比较
  #pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = x # 缓冲输入进来的x矩阵
  return out, cache


def relu_backward(dout, cache):
  """
  Computes the backward pass for a layer of rectified linear units (ReLUs).

  Input:
  - dout: Upstream derivatives, of any shape
  - cache: Input x, of same shape as dout

  Returns:
  - dx: Gradient with respect to x
  """
  dx, x = None, cache
  #############################################################################
  # TODO: Implement the ReLU backward pass.                                   #
  #############################################################################
  dx = (x > 0) * dout # z = max(0, a)
  #pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx


def batchnorm_forward(x, gamma, beta, bn_param):
  """
  Forward pass for batch normalization.
  
  During training the sample mean and (uncorrected) sample variance are
  computed from minibatch statistics and used to normalize the incoming data.
  During training we also keep an exponentially decaying running mean of the mean
  and variance of each feature, and these averages are used to normalize data
  at test-time.

  At each timestep we update the running averages for mean and variance using
  an exponential decay based on the momentum parameter:

  running_mean = momentum * running_mean + (1 - momentum) * sample_mean
  running_var = momentum * running_var + (1 - momentum) * sample_var

  Note that the batch normalization paper suggests a different test-time
  behavior: they compute sample mean and variance for each feature using a
  large number of training images rather than using a running average. For
  this implementation we have chosen to use running averages instead since
  they do not require an additional estimation step; the torch7 implementation
  of batch normalization also uses running averages.

  Input:
  - x: Data of shape (N, D)
  - gamma: Scale parameter of shape (D,)
  - beta: Shift paremeter of shape (D,)
  - bn_param: Dictionary with the following keys:
    - mode: 'train' or 'test'; required
    - eps: Constant for numeric stability
    - momentum: Constant for running mean / variance.
    - running_mean: Array of shape (D,) giving running mean of features
    - running_var Array of shape (D,) giving running variance of features

  Returns a tuple of:
  - out: of shape (N, D)
  - cache: A tuple of values needed in the backward pass
  """
  mode = bn_param['mode']
  eps = bn_param.get('eps', 1e-5) # 数值稳定参数
  momentum = bn_param.get('momentum', 0.9)

  N, D = x.shape
  running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
  running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

  out, cache = None, None
  if mode == 'train':# 训练模式
    #############################################################################
    # TODO: Implement the training-time forward pass for batch normalization.   #
    # Use minibatch statistics to compute the mean and variance, use these      #
    # statistics to normalize the incoming data, and scale and shift the        #
    # normalized data using gamma and beta.                                     #
    #                                                                           #
    # You should store the output in the variable out. Any intermediates that   #
    # you need for the backward pass should be stored in the cache variable.    #
    #                                                                           #
    # You should also use your computed sample mean and variance together with  #
    # the momentum variable to update the running mean and running variance,    #
    # storing your result in the running_mean and running_var variables.        #
    #############################################################################
    sample_mean = np.mean(x, axis = 0) # 每个特征的均值
    sample_var = np.var(x, axis = 0) # 每个特征的方差
    x_hat = (x - sample_mean) / (np.sqrt(sample_var + eps)) # +eps防止分母为0，保持数值稳定

    out = gamma * x_hat + beta

    cache = (x, sample_mean, sample_var, x_hat, eps, gamma, beta)
    running_mean = momentum * running_mean + (1 - momentum) * sample_mean # 动量更新法更新running_mean参数
    # 指数平滑法：最终测试用的running_mean, running_var参数不再是一个bn层决定的，而是所有
    # BN层一起决定
    running_var = momentum * running_var + (1 - momentum) * sample_var
    #pass
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
  elif mode == 'test':# 测试模式
    #############################################################################
    # TODO: Implement the test-time forward pass for batch normalization. Use   #
    # the running mean and variance to normalize the incoming data, then scale  #
    # and shift the normalized data using gamma and beta. Store the result in   #
    # the out variable.                                                         #
    #############################################################################
    out = gamma * (x - running_mean) / (np.sqrt(running_var + eps)) + beta
    #pass
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
  else:
    raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

  # Store the updated running means back into bn_param
  bn_param['running_mean'] = running_mean
  bn_param['running_var'] = running_var

  return out, cache


def batchnorm_backward(dout, cache):
  """
  Backward pass for batch normalization.
  
  For this implementation, you should write out a computation graph for
  batch normalization on paper and propagate gradients backward through
  intermediate nodes.
  
  Inputs:
  - dout: Upstream derivatives, of shape (N, D)
  - cache: Variable of intermediates from batchnorm_forward.
  
  Returns a tuple of:
  - dx: Gradient with respect to inputs x, of shape (N, D)
  - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
  - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
  """
  dx, dgamma, dbeta = None, None, None
  #############################################################################
  # TODO: Implement the backward pass for batch normalization. Store the      #
  # results in the dx, dgamma, and dbeta variables.                           #
  #############################################################################
  x, mean, var, x_hat, eps, gamma, beta = cache
  N = x.shape[0]
  dgamma = np.sum(dout * x_hat, axis = 0) # 见印象笔记BN层反向传播第5行公式
  dbeta = np.sum(dout * 1.0, axis = 0) # 第6行公式
  dx_hat = dout * gamma # 第1行
  dx_hat_numerator = dx_hat / np.sqrt(var + eps) # 第3行第1项
  dx_hat_denominator = np.sum(dx_hat * (x - mean), axis = 0) # 第2行前半部分
  dx_1 = dx_hat_numerator # 第4行第1项
  dvar = dx_hat_denominator * (-0.5) * ((var + eps)**(-1.5)) #第2行
  dmean = -1.0 * np.sum(dx_hat_numerator, axis = 0) + dvar * np.mean((-2.0) * (x - mean) / N, axis = 0) # 第3行
  dx_var = dvar * 2.0 * (x - mean) / N # 第4行第2部分 
  dx_mean = dmean * 1.0 / N # 第4行第3部分 
  dx = dx_1 + dx_var + dx_mean # 第4行

  # gamma, x, u_b, sigma_squared_b, eps, x_hat = cache
  # N = x.shape[0]

  # dx_1 = gamma * dout
  # dx_2_b = np.sum((x - u_b) * dx_1, axis=0)
  # dx_2_a = ((sigma_squared_b + eps) ** -0.5) * dx_1
  # dx_3_b = (-0.5) * ((sigma_squared_b + eps) ** -1.5) * dx_2_b
  # dx_4_b = dx_3_b * 1
  # dx_5_b = np.ones_like(x) / N * dx_4_b
  # dx_6_b = 2 * (x - u_b) * dx_5_b
  # dx_7_a = dx_6_b * 1 + dx_2_a * 1
  # dx_7_b = dx_6_b * 1 + dx_2_a * 1
  # dx_8_b = -1 * np.sum(dx_7_b, axis=0)
  # dx_9_b = np.ones_like(x) / N * dx_8_b
  # dx_10 = dx_9_b + dx_7_a

  # dgamma = np.sum(x_hat * dout, axis=0)
  # dbeta = np.sum(dout, axis=0)
  # dx = dx_10
  #pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
  """
  Alternative backward pass for batch normalization.
  
  For this implementation you should work out the derivatives for the batch
  normalizaton backward pass on paper and simplify as much as possible. You
  should be able to derive a simple expression for the backward pass.
  
  Note: This implementation should expect to receive the same cache variable
  as batchnorm_backward, but might not use all of the values in the cache.
  
  Inputs / outputs: Same as batchnorm_backward
  """
  dx, dgamma, dbeta = None, None, None
  #############################################################################
  # TODO: Implement the backward pass for batch normalization. Store the      #
  # results in the dx, dgamma, and dbeta variables.                           #
  #                                                                           #
  # After computing the gradient with respect to the centered inputs, you     #
  # should be able to compute gradients with respect to the inputs in a       #
  # single statement; our implementation fits on a single 80-character line.  #
  #############################################################################
  gamma, x, sample_mean, sample_var, eps, x_hat = cache
  N = x.shape[0]
  dx_hat = dout * gamma
  dvar = np.sum(dx_hat* (x - sample_mean) * -0.5 * np.power(sample_var + eps, -1.5), axis = 0)
  dmean = np.sum(dx_hat * -1 / np.sqrt(sample_var +eps), axis = 0) + dvar * np.mean(-2 * (x - sample_mean), axis =0)
  dx = 1 / np.sqrt(sample_var + eps) * dx_hat + dvar * 2.0 / N * (x-sample_mean) + 1.0 / N * dmean
  dgamma = np.sum(x_hat * dout, axis = 0)
  dbeta = np.sum(dout , axis = 0)
  #pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  
  return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
  """
  Performs the forward pass for (inverted) dropout.

  Inputs:
  - x: Input data, of any shape
  - dropout_param: A dictionary with the following keys:
    - p: Dropout parameter. We drop each neuron output with probability p.
    - mode: 'test' or 'train'. If the mode is train, then perform dropout;
      if the mode is test, then just return the input.
    - seed: Seed for the random number generator. Passing seed makes this
      function deterministic, which is needed for gradient checking but not in
      real networks.

  Outputs:
  - out: Array of the same shape as x.
  - cache: A tuple (dropout_param, mask). In training mode, mask is the dropout
    mask that was used to multiply the input; in test mode, mask is None.
  """
  p, mode = dropout_param['p'], dropout_param['mode']
  if 'seed' in dropout_param:
    np.random.seed(dropout_param['seed'])# 如果参数字典中有随机数生成种子，则失活罩的随机使用该随机性

  mask = None
  out = None

  if mode == 'train':# 训练模式下： 打开遮活罩
    ###########################################################################
    # TODO: Implement the training phase forward pass for inverted dropout.   #
    # Store the dropout mask in the mask variable.                            #
    ###########################################################################
    keep_prob = 1 - p
    mask = (np.random.randn(*x.shape) < keep_prob) / keep_prob# 训练的时候多除上一个keep_prob
    #测试的时候就不需要除上这个keep_prob
    out = x * mask
    #pass
    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################
  elif mode == 'test':# 测试模式： 关闭失活罩
    ###########################################################################
    # TODO: Implement the test phase forward pass for inverted dropout.       #
    ###########################################################################
    out = x# 不需要除上keep_prob，直接通过dropout层
    #pass
    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################

  cache = (dropout_param, mask)
  out = out.astype(x.dtype, copy=False)

  return out, cache


def dropout_backward(dout, cache):
  """
  Perform the backward pass for (inverted) dropout.

  Inputs:
  - dout: Upstream derivatives, of any shape
  - cache: (dropout_param, mask) from dropout_forward.
  """
  dropout_param, mask = cache
  mode = dropout_param['mode']
  
  dx = None
  if mode == 'train':
    ###########################################################################
    # TODO: Implement the training phase backward pass for inverted dropout.  #
    ###########################################################################
    dx = dout * mask
    #pass
    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################
  elif mode == 'test':
    dx = dout
  return dx


def conv_forward_naive(x, w, b, conv_param):
  """
  A naive implementation of the forward pass for a convolutional layer.

  The input consists of N data points, each with C channels, height H and width
  W. We convolve each input with F different filters, where each filter spans
  all C channels and has height HH and width HH.

  Input:
  - x: Input data of shape (N, C, H, W)
  - w: Filter weights of shape (F, C, HH, WW)
  - b: Biases, of shape (F,)
  - conv_param: A dictionary with the following keys:
    - 'stride': The number of pixels between adjacent receptive fields in the
      horizontal and vertical directions.
    - 'pad': The number of pixels that will be used to zero-pad the input.

  Returns a tuple of:
  - out: Output data, of shape (N, F, H', W') where H' and W' are given by
    H' = 1 + (H + 2 * pad - HH) / stride
    W' = 1 + (W + 2 * pad - WW) / stride
  - cache: (x, w, b, conv_param)
  """
  out = None
  #############################################################################
  # TODO: Implement the convolutional forward pass.                           #
  # Hint: you can use the function np.pad for padding.                        #
  #############################################################################
  #任务：完成带卷机操作的前向操作
  #提示，可以使用Np.pad函数实现padding操作
  N, C, H, W = x.shape # N个样本，C个通道，H的高度，W的宽度
  F, C, HH, WW = w.shape # F个滤波器，C个通道，HH的滤波器高度，WW的滤波器宽度
  stride = conv_param['stride'] # 滤波器每次移动的步长
  pad = conv_param['pad'] # 图片填充的宽度

  # 计算卷积结果矩阵的大小并分配全零值占位
  new_H = 1 + int((H + 2 * pad - HH) / stride)
  new_W = 1 + int((W + 2 * pad - WW) / stride)
  out = np.zeros((N, F, new_H, new_W))

  # 卷积开始
  for n in range(N):
    for f in range(F):
      # 临时分配（new_H, new_W)大小的全偏移项卷积矩阵，（即提前加上偏移项b[f]）
      conv_newH_newW = np.ones((new_H, new_W)) * b[f]
      for c in range(C):
        # 填充原始矩阵，填充大小为pad，填充值为0
        padded_x = np.lib.pad(x[n, c], pad_width = pad, mode = 'constant', constant_values = 0)
        for i in range(new_H): # 对每个通道的一个样本数据矩阵的卷积矩阵X操作
          for j in range(new_W):
            conv_newH_newW[i, j] += np.sum(padded_x[i * stride: i * stride + HH, j * stride: j * stride + WW] * w[f, c, :, :]) # new_x = x * w + b
        out[n, f] = conv_newH_newW
  #pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x, w, b, conv_param)
  return out, cache


def conv_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a convolutional layer.

  Inputs:
  - dout: Upstream derivatives.
  - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

  Returns a tuple of:
  - dx: Gradient with respect to x
  - dw: Gradient with respect to w
  - db: Gradient with respect to b
  """
  dx, dw, db = None, None, None
  #############################################################################
  # TODO: Implement the convolutional backward pass.                          #
  #############################################################################
  # 任务：卷积层的反向传播
  # 数据准备
  x, w, b, conv_param = cache
  pad = conv_param['pad']
  stride = conv_param['stride']
  F, C, HH, WW = w.shape
  N, C, H, W = x.shape
  N, F, new_H, new_W = dout.shape

  # 下面，我们模拟卷积，首先填充x。
  padded_x = np.lib.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode = 'constant', constant_values = 0)
  padded_dx = np.zeros_like(padded_x)# 填充了的dx，后面去填充即可得到dx
  dw = np.zeros_like(w)
  db = np.zeros_like(b)

  for n in range(N): # 第n个图像
    for f in range(F): # 第f个滤波器
      for i in range(new_H):
        for j in range(new_W):
          db[f] += dout[n, f, i, j] # dg对db求导:1*dout
          dw[f] += padded_x[n, :, i * stride: i * stride + HH, j * stride: j * stride + WW] * dout[n, f, i, j]
          padded_dx[n, :, i * stride: i * stride + HH, j * stride: j * stride + WW] += w[f] * dout[n, f, i, j]
  #去掉填充部分
  dx = padded_dx[:, :, pad:pad + H, pad:pad + W]
  #pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx, dw, db


def max_pool_forward_naive(x, pool_param):
  """
  A naive implementation of the forward pass for a max pooling layer.

  Inputs:
  - x: Input data, of shape (N, C, H, W)
  - pool_param: dictionary with the following keys:
    - 'pool_height': The height of each pooling region
    - 'pool_width': The width of each pooling region
    - 'stride': The distance between adjacent pooling regions

  Returns a tuple of:
  - out: Output data
  - cache: (x, pool_param)
  """
  out = None
  #############################################################################
  # TODO: Implement the max pooling forward pass                              #
  #############################################################################
  # 任务：实现正向最大池化操作
  N, C, H, W = x.shape
  HH, WW, stride = pool_param['pool_height'], pool_param['pool_width'], pool_param['stride']
  H_out = (H-HH)/stride+1
  W_out = (W-WW)/stride+1
  out = np.zeros((N,C,H_out,W_out))
  for i in xrange(H_out):
        for j in xrange(W_out):
            x_masked = x[:,:,i*stride : i*stride+HH, j*stride : j*stride+WW]
            out[:,:,i,j] = np.max(x_masked, axis=(2,3))

  # N, C, H, W = x.shape
  # pool_height = pool_param['pool_height']
  # pool_width = pool_param['pool_width']
  # pool_stride = pool_param['stride']

  # new_H = 1 + int((H - pool_height / pool_stride))
  # new_W = 1 + int((W - pool_width / pool_stride))

  # out = np.zeros((N, C, new_H, new_W))

  # for n in range(N):
  #   for c in range(C):
  #     for i in range(new_H):
  #       for j in range(new_W):
  #         out[n, c, i, j] = np.max(x[n, c, i * pool_stride:i * pool_stride + pool_height, j * pool_stride:j * pool_stride + pool_width])
  #pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x, pool_param)
  return out, cache


def max_pool_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a max pooling layer.

  Inputs:
  - dout: Upstream derivatives
  - cache: A tuple of (x, pool_param) as in the forward pass.

  Returns:
  - dx: Gradient with respect to x
  """
  dx = None
  #############################################################################
  # TODO: Implement the max pooling backward pass                             #
  #############################################################################
  # 反向最大池化操作
  # 池化层没有参数，只需求出当前层的梯度dx
  x, pool_param = cache
  N, C, H, W = x.shape
  pool_height = pool_param['pool_height']
  pool_width = pool_param['pool_width']
  pool_stride = pool_param['stride']

  new_H = 1 + int((H - pool_height) / pool_stride)
  new_W = 1 + int((W - pool_width) / pool_stride)
  dx = np.zeros_like(x)

  for n in range(N):
    for c in range(C):
      for i in range(new_H):
        for j in range(new_W):
          window = x[n, c, i * pool_stride:i * pool_stride + pool_height, j * pool_stride:j * pool_stride + pool_width]
          dx[n, c, i * pool_stride:i * pool_stride + pool_height, j * pool_stride:j * pool_stride + pool_width] = (window == np.max(window)) * dout[n, c, i, j]
          # 找到当前池化窗口中最大的数值 还原
  #pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
  """
  Computes the forward pass for spatial batch normalization.
  空间批量归一化
  Inputs:
  - x: Input data of shape (N, C, H, W)
  - gamma: Scale parameter, of shape (C,)
  - beta: Shift parameter, of shape (C,)
  - bn_param: Dictionary with the following keys:
    - mode: 'train' or 'test'; required
    - eps: Constant for numeric stability
    - momentum: Constant for running mean / variance. momentum=0 means that
      old information is discarded completely at every time step, while
      momentum=1 means that new information is never incorporated. The
      default of momentum=0.9 should work well in most situations.
    - running_mean: Array of shape (D,) giving running mean of features
    - running_var Array of shape (D,) giving running variance of features
    
  Returns a tuple of:
  - out: Output data, of shape (N, C, H, W)
  - cache: Values needed for the backward pass
  """
  out, cache = None, None

  #############################################################################
  # TODO: Implement the forward pass for spatial batch normalization.         #
  #                                                                           #
  # HINT: You can implement spatial batch normalization using the vanilla     #
  # version of batch normalization defined above. Your implementation should  #
  # be very short; ours is less than five lines.                              #
  #############################################################################
  N, C, H, W = x.shape
  x_new = x.transpose(0, 2, 3, 1).reshape((N * H * W, C)) # 重新构造数据矩阵
  # 将每个通道的图片分开来归一化
  out, cache = batchnorm_forward(x_new, gamma, beta, bn_param)
  # 重新构造归一化好的输出矩阵
  out = out.reshape(N, H, W, C).transpose(0, 3, 1, 2)
  #pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return out, cache


def spatial_batchnorm_backward(dout, cache):
  """
  Computes the backward pass for spatial batch normalization.
  
  Inputs:
  - dout: Upstream derivatives, of shape (N, C, H, W)
  - cache: Values from the forward pass
  
  Returns a tuple of:
  - dx: Gradient with respect to inputs, of shape (N, C, H, W)
  - dgamma: Gradient with respect to scale parameter, of shape (C,)
  - dbeta: Gradient with respect to shift parameter, of shape (C,)
  """
  dx, dgamma, dbeta = None, None, None

  #############################################################################
  # TODO: Implement the backward pass for spatial batch normalization.        #
  #                                                                           #
  # HINT: You can implement spatial batch normalization using the vanilla     #
  # version of batch normalization defined above. Your implementation should  #
  # be very short; ours is less than five lines.                              #
  #############################################################################
  N, C, H, W = dout.shape
  dout_new = dout.transpose(0, 2, 3, 1).reshape((N * H * W, C)) # 重新构造数据矩阵
  dx, dgamma, dbeta = batchnorm_bacayerrkward(dout_new, cache)
  # 重新构造归一化好的输出矩阵
  dx = dx.reshape(N, H, W, C).transpose(0, 3, 1, 2)
  #pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return dx, dgamma, dbeta
  

def svm_loss(x, y):
  """
  Computes the loss and gradient using for multiclass SVM classification.

  Inputs:
  - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
    for the ith input.
  - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
    0 <= y[i] < C

  Returns a tuple of:
  - loss: Scalar giving the loss
  - dx: Gradient of the loss with respect to x
  """
  N = x.shape[0]
  correct_class_scores = x[np.arange(N), y]
  margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
  margins[np.arange(N), y] = 0
  loss = np.sum(margins) / N
  num_pos = np.sum(margins > 0, axis=1)
  dx = np.zeros_like(x)
  dx[margins > 0] = 1
  dx[np.arange(N), y] -= num_pos
  dx /= N
  return loss, dx


def softmax_loss(x, y):
  """
  Computes the loss and gradient for softmax classification.

  Inputs:
  - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
    for the ith input.
  - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
    0 <= y[i] < C

  Returns a tuple of:
  - loss: Scalar giving the loss
  - dx: Gradient of the loss with respect to x
  """
  probs = np.exp(x - np.max(x, axis=1, keepdims=True))
  probs /= np.sum(probs, axis=1, keepdims=True)
  N = x.shape[0]
  loss = -np.sum(np.log(probs[np.arange(N), y])) / N
  dx = probs.copy()
  dx[np.arange(N), y] -= 1
  dx /= N
  return loss, dx
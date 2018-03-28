# -*- coding: UTF-8 -*-
import numpy as np

from cs231n.layers import *
from cs231n.layer_utils import *


class TwoLayerNet(object):
  """
  A two-layer fully-connected neural network with ReLU nonlinearity and
  softmax loss that uses a modular layer design. We assume an input dimension
  of D, a hidden dimension of H, and perform classification over C classes.
  
  The architecure should be affine - relu - affine - softmax.

  Note that this class does not implement gradient descent; instead, it
  will interact with a separate Solver object that is responsible for running
  optimization.

  The learnable parameters of the model are stored in the dictionary
  self.params that maps parameter names to numpy arrays.
  """
  # 初始化神经网络
  def __init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10,
               weight_scale=1e-3, reg=0.0):
    """
    Initialize a new network.

    Inputs:
    - input_dim: An integer giving the size of the input
    - hidden_dim: An integer giving the size of the hidden layer
    - num_classes: An integer giving the number of classes to classify
    - dropout: Scalar between 0 and 1 giving dropout strength.
    - weight_scale: Scalar giving the standard deviation for random
      initialization of the weights.
    - reg: Scalar giving L2 regularization strength.
    """
    self.params = {}
    self.reg = reg
    
    ############################################################################
    # TODO: Initialize the weights and biases of the two-layer net. Weights    #
    # should be initialized from a Gaussian with standard deviation equal to   #
    # weight_scale, and biases should be initialized to zero. All weights and  #
    # biases should be stored in the dictionary self.params, with first layer  #
    # weights and biases using the keys 'W1' and 'b1' and second layer weights #
    # and biases using the keys 'W2' and 'b2'.                                 #
    ############################################################################
    self.params['W1'] = weight_scale * np.random.randn(input_dim, hidden_dim)
    self.params['b1'] = np.zeros(hidden_dim)
    self.params['W2'] = weight_scale * np.random.randn(hidden_dim, num_classes)
    self.params['b2'] = np.zeros(num_classes)
    #pass
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

  # 接下来定义损失函数完成神经网络的构造  
  def loss(self, X, y=None):
    """
    Compute loss and gradient for a minibatch of data.

    Inputs:
    - X: Array of input data of shape (N, d_1, ..., d_k)
    - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

    Returns:
    If y is None, then run a test-time forward pass of the model and return:
    - scores: Array of shape (N, C) giving classification scores, where
      scores[i, c] is the classification score for X[i] and class c.

    If y is not None, then run a training-time forward and backward pass and
    return a tuple of:
    - loss: Scalar value giving the loss
    - grads: Dictionary with the same keys as self.params, mapping parameter
      names to gradients of the loss with respect to those parameters.
    """  
    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the two-layer net, computing the    #
    # class scores for X and storing them in the scores variable.              #
    ############################################################################
    #a1_out, a1_cache = affine_forward(X, self.params['W1'], self.params['b1'])
    #r1_out, r1_cache = relu_forward(a1_out)
    #数据在隐藏层和输出层的前向传播：
    h1_out, h1_cache = affine_relu_forward(X, self.params['W1'], self.params['b1'])
    scores, out_cache = affine_forward(h1_out, self.params['W2'], self.params['b2'])
    #pass
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    # If y is None then we are in test mode so just return scores
    if y is None:
      return scores
    
    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the two-layer net. Store the loss  #
    # in the loss variable and gradients in the grads dictionary. Compute data #
    # loss using softmax, and make sure that grads[k] holds the gradients for  #
    # self.params[k]. Don't forget to add L2 regularization!                   #
    #                                                                          #
    # NOTE: To ensure that your implementation matches ours and you pass the   #
    # automated tests, make sure that your L2 regularization includes a factor #
    # of 0.5 to simplify the expression for the gradient.                      #
    ############################################################################
    #输出层后，结合正确标签y得出损失值和其在输出层的梯度：
    loss, dout = softmax_loss(scores, y)
    loss += 0.5 * self.reg * (np.sum(self.params['W1'] * self.params['W1']) + np.sum(self.params['W2'] * self.params['W2'])) # 加入正则化损失

    dout, dw2, db2 = affine_backward(dout, out_cache)
    grads['W2'] = dw2 + self.reg * self.params['W2']
    grads['b2'] = db2
    _, dw1, db1 = affine_relu_backward(dout, h1_cache)
    grads['W1'] = dw1 + self.reg * self.params['W1']
    grads['b1'] = db1

    #pass
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return loss, grads


class FullyConnectedNet(object):
  """# 构造任意深度的神经网络
  A fully-connected neural network with an arbitrary number of hidden layers,
  ReLU nonlinearities, and a softmax loss function. This will also implement
  dropout and batch normalization as options. For a network with L layers,
  the architecture will be
  
  {affine - [batch norm] - relu - [dropout]} x (L - 1) - affine - softmax
  
  where batch normalization and dropout are optional, and the {...} block is
  repeated L - 1 times.
  
  Similar to the TwoLayerNet above, learnable parameters are stored in the
  self.params dictionary and will be learned using the Solver class.
  """
  """
    一个任意隐藏层数和神经元数的全连接神经网络，其中 ReLU 激活函数，sofmax 损失函数，同时可选的采用 dropout 和 batch normalization（批量归一化）。那么，对于一个 L 层的神经网络来说，其框架是：

    {affine -  [batch norm] - relu 一【dropout] } x  (L 一 1) - affine - softmax 
    其中的【batch norm】和【dropout】是可选非必须的，框架中{... }部分将会重复 L-1 次，代表 L-1 个隐藏层。

    与我们在上面的故事中定义的 TwoLayerNet（）类保持一致，所有待学习的参数都会存在

    self. Params 字典中，并且最终会被最优化 Solver（）类训练学习得到（后面的故事会谈到）。
  """
  # 第一步：初始化神经网络（参数）
  def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
               dropout=0, use_batchnorm=False, reg=0.0,
               weight_scale=1e-2, dtype=np.float32, seed=None):

    """
    hidden_dims:一个列表，元素个数是隐藏层数，元素值为该层神经元数

    input_dim:默认输入神经元的个数是 3072 个（匹配 CIFAR-10 数据集）

    num_classes:默认输出神经元的个数是 10 个（匹配 CIFAR-10 数据集）

    dropout:默认不开启 dropout，若取（0, 1) 表示失活概率

    use_batchnorm:默认不开启批量归一化，若开启取 True 
    reg:默认无 L2 正则化，取某 scalar 表示正则化的强度
    weight_scale:默认 0.01, 表示权重参数初始化的标准差

    dtype:默认 np. Float64 精度，要求所有的计算都应该在此精度下。
    seed:默认无随机种子，若有会传递给 dropout 层。
    """
    """
    Initialize a new FullyConnectedNet.
    
    Inputs:
    - hidden_dims: A list of integers giving the size of each hidden layer.
    - input_dim: An integer giving the size of the input.
    - num_classes: An integer giving the number of classes to classify.
    - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=0 then
      the network should not use dropout at all.
    - use_batchnorm: Whether or not the network should use batch normalization.
    - reg: Scalar giving L2 regularization strength.
    - weight_scale: Scalar giving the standard deviation for random
      initialization of the weights.
    - dtype: A numpy datatype object; all computations will be performed using
      this datatype. float32 is faster but less accurate, so you should use
      float64 for numeric gradient checking.
    - seed: If not None, then pass this random seed to the dropout layers. This
      will make the dropout layers deteriminstic so we can gradient check the
      model.
    """
    self.use_batchnorm = use_batchnorm
    self.use_dropout = dropout > 0
    self.reg = reg
    self.num_layers = 1 + len(hidden_dims) # 神经网络的层数
    self.dtype = dtype
    self.params = {}

    ############################################################################
    # TODO: Initialize the parameters of the network, storing all values in    #
    # the self.params dictionary. Store weights and biases for the first layer #
    # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
    # initialized from a normal distribution with standard deviation equal to  #
    # weight_scale and biases should be initialized to zero.                   #
    #                                                                          #
    # When using batch normalization, store scale and shift parameters for the #
    # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
    # beta2, etc. Scale parameters should be initialized to one and shift      #
    # parameters should be initialized to zero.                                #
    ############################################################################
    # 定义所有隐藏层的参数到字典self.params中：
    in_dim = input_dim # in_dim = D

    for i, h_dim in enumerate(hidden_dims): # Eg:(i, h_dim) = (0,H1),(1,H2)...
        # Eg:W1(D, H1),W2(H1, H2)...
        self.params['W%d' %(i + 1,)] = weight_scale * np.random.randn(in_dim, h_dim)
        # 小随机数为初始值
        # Eg:b1(H1,), b2(H2)...
        self.params['b%d' %(i + 1,)] = np.zeros((h_dim,))
        # 0为初始值
        if use_batchnorm: # 如果使用了BN层
            #Eg: gamma1(H1), gamma2(H2)... 1为初始值
            #Eg: beta(H1), beta2(H2)... 0为初始值
            self.params['gamma%d' %(i + 1,)] = np.ones(h_dim)
            self.params['beta%d' %(i + 1,)] = np.zeros(h_dim)
        in_dim = h_dim # 将该隐藏层的层数传递给下一层的行数
    #pass
    #定义输出层的参数到字典params中：
    self.params['W%d' % (self.num_layers,)] = weight_scale * np.random.randn(in_dim, num_classes)
    self.params['b%d' % (self.num_layers,)] = np.zeros((num_classes,))
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    # When using dropout we need to pass a dropout_param dictionary to each
    # dropout layer so that the layer knows the dropout probability and the mode
    # (train / test). You can pass the same dropout_param to each dropout layer.

    # 当开启dropout时，我们需要在每一个神经元层中传递一个相同的dropout参数字典self.dropout_param，以保证
    # 每一层的神经元们都知晓失活概率p和当前神经网络的模式状态mode（训练/测试）
    self.dropout_param = {} # dropout的参数字典
    if self.use_dropout: # 如果use_dropout的值时（0，1），即启用drropout
      self.dropout_param = {'mode': 'train', 'p': dropout}
      if seed is not None:# 如果有随机种子，存入seed
        self.dropout_param['seed'] = seed 
    
    # With batch normalization we need to keep track of running means and
    # variances, so we need to pass a special bn_param object to each batch
    # normalization layer. You should pass self.bn_params[0] to the forward pass
    # of the first batch normalization layer, self.bn_params[1] to the forward
    # pass of the second batch normalization layer, etc.

    # 开启批量归一化时，我们要定义一个BN算法的参数列表self.bn_params,以用来跟踪记录每一层的平均值和标准差。
    # 其中，第0个元素self.bn_params[0]表示前向传播第1个BN层的参数，第1个元素self.bn_params[1] 表示前向传播第2个BN层的参数，以此类推
    self.bn_params = [] # BN算法的参数列表
    if self.use_batchnorm: # 如果开启归一化， 设置每层mode默认为训练模式
      self.bn_params = [{'mode': 'train'} for i in xrange(self.num_layers - 1)]
        # 上面self.bn_params 列表个数时hidden layers的个数
    # Cast all parameters to the correct datatype
    # 最后，调整所有待学习神经网络参数为制定计算精度：np.float64
    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)

  # 第二步：定义我们的损失函数    
  def loss(self, X, y=None):
    """
    和 TwoLayerNet（）一样：

    首先，输入的数据 x 是一个多维的 array，shape 为（样本图片的个数 N*3*32*32), y 是与输入数据 X 对应的正确标签，shape 为（N，）。
    #在训练模式下：#

    我们 loss 函数目标输出一个损失值 loss 和一个 grads 字典，

    其中存有 loss 关于隐藏层和输出层的参数（W, B, gamma, beta）的梯度值。
    #在测试模式下：#

    我们的 loss 函数只需要直接给出输出层后的得分即可。
    """
    """
    Compute loss and gradient for the fully-connected net.

    Input / output: Same as TwoLayerNet above.
    """
    # 把输入数据源矩阵的X的精度调整一下
    X = X.astype(self.dtype)
    # 根据正确标签y是否为None来调整模式是Test还是Train
    mode = 'test' if y is None else 'train'

    # Set train/test mode for batchnorm params and dropout param since they
    # behave differently during training and testing.

    # 确定了当前神经网络所处的模式状态后，就可以设置 dropout 的参数字典和 BN 算法的参数列表中的 mode 了，因为他们在不同的模式下行为是不同的。
    if self.dropout_param is not None:# 如果开启dropout
      self.dropout_param['mode'] = mode   
    if self.use_batchnorm: # 如果开启批量归一化
      for bn_param in self.bn_params:
        bn_param['mode'] = mode

    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the fully-connected net, computing  #
    # the class scores for X and storing them in the scores variable.          #
    #                                                                          #
    # When using dropout, you'll need to pass self.dropout_param to each       #
    # dropout forward pass.                                                    #
    #                                                                          #
    # When using batch normalization, you'll need to pass self.bn_params[0] to #
    # the forward pass for the first batch normalization layer, pass           #
    # self.bn_params[1] to the forward pass for the second batch normalization #
    # layer, etc.                                                              #
    ############################################################################
    """
    %前向传播%

    如果开启了 dropout，我们需要将 dropout 的参数字典 self. Dropout_ param 在每一个 dropout 层中传递。

    如果开启了批量归一化，我们需要指定 BN 算法的参数列表 self. Bn_ params [0] 对应前向传播第一层的参数，self. Bn_ params [1] 对应第二层的参数，以此类推。
    """
    fc_mix_cache = {} # 初始化每层前向传播的缓冲字典
    if self.use_dropout: # 如果开启了dropout，初始化其对应的缓冲字典
        dp_cache = {}

    # 从第一个隐藏层开始循环每一个隐藏层，传递数据out，保存每一层中的缓冲cache
    out = X
    for i in range(self.num_layers - 1): # 在每个hidden层中循环
        w, b = self.params['W%d' %(i + 1,)], self.params['b%d' %(i + 1,)]
        if self.use_dropout: # 若开启批量归一化
            gamma = self.params['gamma%d' %(i + 1,)]
            beta = self.params['beta%d' %(i + 1,)]
            out, fc_mix_cache[i] = affine_bn_relu_forward(out, w, b, gamma, beta, self.bn_params[i])
        else:
            out, fc_mix_cache[i] = affine_relu_forward(out, w, b)

        if self.use_dropout:
            out, dp_cache[i] = dropout_forward(out, self.dropout_param)

    # 最后的输出层        
    w = self.params['W%d' % (self.num_layers,)]
    b = self.params['b%d' % (self.num_layers,)]
    out, out_cache = affine_forward(out, w, b)
    scores = out

    """
    可以看到，上面对隐藏层的每次循环中，out 变量实现了自我迭代更新；

    fc_ mix_ cache 缓冲字典中顺序地存储了每个隐藏层的得分情况和模型参数（其中可内含 BN 层）; 
    dp_ cache 缓冲字典中单独顺序地保存了每个 dropout 层的失活概率和遮罩；
    out_ cache 变量缓存了输出层处的信息；

    值得留意的是，若开启批量归一化的话，BN 层的参数列表 self. Bn_ params [i],在bn_forward函数中

    从第一层开始多出'running_ mean '和'running_ var'的键值保存在参数列表的每一个元素中，形如： [{ 'mode': 'train', 'running_ mean': ***, 'running_ var': ***}, {... }]
    """
    #pass
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    # If test mode return early
    if mode == 'test': # 如果是测试模式，输出scores表示预测的每个分类分数后，函数停止，跳出
      return scores

    loss, grads = 0.0, {}
    ############################################################################
    # TODO: Implement the backward pass for the fully-connected net. Store the #
    # loss in the loss variable and gradients in the grads dictionary. Compute #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    #                                                                          #
    # When using batch normalization, you don't need to regularize the scale   #
    # and shift parameters.                                                    #
    #                                                                          #
    # NOTE: To ensure that your implementation matches ours and you pass the   #
    # automated tests, make sure that your L2 regularization includes a factor #
    # of 0.5 to simplify the expression for the gradient.                      #
    ############################################################################
    # 开始反向传播计算梯度
    loss, dout = softmax_loss(scores, y)
    loss += 0.5 * self.reg * np.sum(self.params['W%d' %(self.num_layers,)]**2)

    # 输出层处梯度的反向传播，顺便把梯度保存在梯度字典grad中：
    dout, dw, db = affine_backward(dout, out_cache)
    grads['W%d' %(self.num_layers,)] = dw + self.reg * self.params['W%d' %(self.num_layers,)]
    grads['b%d' %(self.num_layers,)] = db + self.reg * self.params['b%d' %(self.num_layers,)]

    # 在每一个隐藏层处的梯度的方向传播，不仅顺便更新了梯度字典grad,还迭代算出了损失值loss；
    for i in range(self.num_layers - 1):
        ri = self.num_layers - 2 - i # 倒数第ri+1隐藏层
        loss += 0.5 * self.reg * np.sum(self.params['W%d' %(ri + 1,)]**2) # 迭代的补上正则项

        if self.use_dropout: # 若开启dropout
            dout = dropout_backward(dout, dp_cache[ri])

        if self.use_batchnorm:# 若开启批量归一化
            dout, dw, db, dgamma, dbeta = affine_bn_relu_backward(dout, fc_mix_cache[ri])
            grads['gamma%d' %(ri + 1,)] =dgamma
            grads['beta%d' %(ri + 1,)] =dbeta
        else:
            dout, dw, db = affine_relu_backward(dout, fc_mix_cache[ri])   

        grads['W%d' %(ri + 1,)] = dw + self.reg * self.params['W%d' %(ri + 1,)]
        grads['b%d' %(ri + 1,)] = db

    #pass
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return loss, grads

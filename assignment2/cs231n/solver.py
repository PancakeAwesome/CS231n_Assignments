# -*- coding: UTF-8 -*-
import numpy as np

from cs231n import optim


class Solver(object):
  """
  A Solver encapsulates all the logic necessary for training classification
  models. The Solver performs stochastic gradient descent using different
  update rules defined in optim.py.

  The solver accepts both training and validataion data and labels so it can
  periodically check classification accuracy on both training and validation
  data to watch out for overfitting.

  To train a model, you will first construct a Solver instance, passing the
  model, dataset, and various optoins (learning rate, batch size, etc) to the
  constructor. You will then call the train() method to run the optimization
  procedure and train the model.
  
  After the train() method returns, model.params will contain the parameters
  that performed best on the validation set over the course of training.
  In addition, the instance variable solver.loss_history will contain a list
  of all losses encountered during training and the instance variables
  solver.train_acc_history and solver.val_acc_history will be lists containing
  the accuracies of the model on the training and validation set at each epoch.
  
  Example usage might look something like this:
  
  data = {
    'X_train': # training data
    'y_train': # training labels
    'X_val': # validation data
    'X_train': # validation labels
  }
  model = MyAwesomeModel(hidden_size=100, reg=10)
  solver = Solver(model, data,
                  update_rule='sgd',
                  optim_config={
                    'learning_rate': 1e-3,
                  },
                  lr_decay=0.95,
                  num_epochs=10, batch_size=100,
                  print_every=100)
  solver.train()


  A Solver works on a model object that must conform to the following API:

  - model.params must be a dictionary mapping string parameter names to numpy
    arrays containing parameter values.

  - model.loss(X, y) must be a function that computes training-time loss and
    gradients, and test-time classification scores, with the following inputs
    and outputs:

    Inputs:
    - X: Array giving a minibatch of input data of shape (N, d_1, ..., d_k)
    - y: Array of labels, of shape (N,) giving labels for X where y[i] is the
      label for X[i].

    Returns:
    If y is None, run a test-time forward pass and return:
    - scores: Array of shape (N, C) giving classification scores for X where
      scores[i, c] gives the score of class c for X[i].

    If y is not None, run a training time forward and backward pass and return
    a tuple of:
    - loss: Scalar giving the loss
    - grads: Dictionary with the same keys as self.params mapping parameter
      names to gradients of the loss with respect to those parameters.
  """

  """
  我们定义的这个 Solver 类将会根据我们的神经网络模型框架一-FullyConnectedNet（）类，

  在数据源的训练集部分和验证集部分中，训练我们的模型，并且通过周期性的检查准确率的方式，以避免过拟合。

  在这个类中，包括_ init_ (），共定义 5 个函数，其中只有 train（）函数是最重要的。调用它后，会自动启动神经网络模型优化程序。

  训练结束后，经过更新在验证集上优化后的模型参数会保存在 model. Params 中。此外，损失值的历史训练信息会保存在 solver.loss_history 中，还有 solver.train_acc_history 和 solver.val_acc_history 中会分别保存训|练集和验证集在每一次 epoch 时的模型准确率。
  下面是给出一个Solver类使用的实例：
  data = {
    'X_train': # training data
    'y_train': # training labels
    'X_val': # validation data
    'X_train': # validation labels
  } # 以字典的形式存入训练集和验证集的数据和标签
  model = MyAwesomeModel(hidden_size=100, reg=10) # 我们的神经网络模型
  solver = Solver(model, data, #模型/数据
                  update_rule='sgd', #优化算法
                  optim_config={ #该优化算法的参数
                    'learning_rate': 1e-3, #学习率
                  },
                  lr_decay=0.95, # 学习率的衰减速率
                  num_epochs=10, # 训练模型的遍数
                  batch_size=100, # 每次丢入模型的图片数目
                  print_every=100) 
  solver.train()
  # 神经模型中必须要有两个函数方法：模型参数model.params和损失函数Model.loss(X, y)
  A Solver works on a model object that must conform to the following API:

  - model.params must be a dictionary mapping string parameter names to numpy
    arrays containing parameter values.

  - model.loss(X, y) must be a function that computes training-time loss and
    gradients, and test-time classification scores, with the following inputs
    and outputs:

    Inputs:  全局的输入变量
    - X: Array giving a minibatch of input data of shape (N, d_1, ..., d_k)
    - y: Array of labels, of shape (N,) giving labels for X where y[i] is the
      label for X[i].

    Returns: # 全局的输出变量
    # 用标签Y的存在与否标记训练mode还是测试mode
    If y is None, run a test-time forward pass and return:
    - scores: Array of shape (N, C) giving classification scores for X where
      scores[i, c] gives the score of class c for X[i].

    If y is not None, run a training time forward and backward pass and return
    a tuple of:
    - loss: Scalar giving the loss # 损失函数值
    - grads: Dictionary with the same keys as self.params mapping parameter
      names to gradients of the loss with respect to those parameters. # 模型梯度
  """

  def __init__(self, model, data, **kwargs):
    """
    #step1#  初始化我们的Solver类：
    Construct a new Solver instance.
    # 必须要输入的函数参数：模型和数据
    Required arguments:
    - model: A model object conforming to the API described above
    - data: A dictionary of training and validation data with the following:
      'X_train': Array of shape (N_train, d_1, ..., d_k) giving training images
      'X_val': Array of shape (N_val, d_1, ..., d_k) giving validation images
      'y_train': Array of shape (N_train,) giving labels for training images
      'y_val': Array of shape (N_val,) giving labels for validation images
    # 可选的输入参数：  
    Optional arguments:
      # 优化算法：默认为sgd
    - update_rule: A string giving the name of an update rule in optim.py.
      Default is 'sgd'.
      设置优化算法的超参数：
    - optim_config: A dictionary containing hyperparameters that will be
      passed to the chosen update rule. Each update rule requires different
      hyperparameters (see optim.py) but all update rules require a
      'learning_rate' parameter so that should always be present.
      学习率在每次epoch时的衰减率
    - lr_decay: A scalar for learning rate decay; after each epoch the learning
      rate is multiplied by this value.
      在训练时，模型输入层接受样本图片的大小，默认100
    - batch_size: Size of minibatches used to compute loss and gradient during
      training.
      在训练时，让神经网络模型一次全套训练的遍数
    - num_epochs: The number of epochs to run for during training.
      在训练时，打印损失值的迭代次数
    - print_every: Integer; training losses will be printed every print_every
      iterations.
      是否在训练时输出中间过程
    - verbose: Boolean; if set to false then no output will be printed during
      training.
    """
    # 实例中增加变量并赋予初值，以方便后面的train()函数等调用：
    # 必选参数：模型和数据
    self.model = model
    self.X_train = data['X_train']
    self.y_train = data['y_train']
    self.X_val = data['X_val']
    self.y_val = data['y_val']
    
    #可选参数：逐渐一个一个剪切打包Kwargs参数列表
    # Unpack keyword arguments
    self.update_rule = kwargs.pop('update_rule', 'sgd')
    self.optim_config = kwargs.pop('optim_config', {})
    self.lr_decay = kwargs.pop('lr_decay', 1.0)
    self.batch_size = kwargs.pop('batch_size', 100)
    self.num_epochs = kwargs.pop('num_epochs', 10)

    self.print_every = kwargs.pop('print_every', 10)
    self.verbose = kwargs.pop('verbose', True)

    # 参数异常处理：如果kwargs参数列表中除了上述元素外还有其他的就报错！
    # Throw an error if there are extra keyword arguments
    if len(kwargs) > 0:
      extra = ', '.join('"%s"' % k for k in kwargs.keys())
      raise ValueError('Unrecognized arguments %s' % extra)

    # 参数异常处理：如果kwargs参数列表中没有优化算法，就报错！
    # 蒋self.update_rule转换为优化算法的函数，即：
    # self.update_rule(w, dw, config) = (next_w, config)

    # Make sure the update rule exists, then replace the string
    # name with the actual function
    if not hasattr(optim, self.update_rule): # 若optim.py中没有写好的优化算法对应
      raise ValueError('Invalid update_rule "%s"' % self.update_rule)
    self.update_rule = getattr(optim, self.update_rule)
    # 执行_reset()函数：
    self._reset()


  def _reset(self):
    """
    #step2#:定义我们的_reset()函数，其仅在类初始化函数_init_()中调用
    主要目的是更新我们的优化算法所需要的参数
    Set up some book-keeping variables for optimization. Don't call this
    manually.

    """
    # 重置一些用于记录优化的变量
    # Set up some variables for book-keeping
    self.epoch = 0
    self.best_val_acc = 0
    self.best_params = {}
    self.loss_history = []
    self.train_acc_history = []
    self.val_acc_history = []

    # Make a deep copy of the optim_config for each parameter
    # 上面根据模型中待学习的参数，创建了新的优化字典self.optim_configs;
    #形如：{'b':{'learning_rate': 0.0005},
    # 'w': {'learning_rate': 0.0005}, 为每个模型参数指定了相同的超参数。
    #}
    
    self.optim_configs = {}
    for p in self.model.params:
      d = {k: v for k, v in self.optim_config.iteritems()}
      self.optim_configs[p] = d

  #3# 定义我们的_step()函数，其仅在train()函数中调用
  def _step(self):
    """
    Make a single gradient update. This is called by train() and should not
    be called manually.
    训练模式下，样本图片数据的一次正向和反向传播，并且更新模型参数一次。
    """
    # Make a minibatch of training data # 输入数据准备
    num_train = self.X_train.shape[0] #要训练的数据集总数
    batch_mask = np.random.choice(num_train, self.batch_size)
    X_batch = self.X_train[batch_mask] # 随机取得输入神经元的样本图片数据
    y_batch = self.y_train[batch_mask] # 随机取得输入神经元的样本图片标签

    # Compute loss and gradient # 数据通过神经网络后得到损失值和梯度字典
    loss, grads = self.model.loss(X_batch, y_batch)
    self.loss_history.append(loss) # 把本次算得的损失值记录下来

    # Perform a parameter update # 执行一次模型参数的更新
    for p, w in self.model.params.iteritems():
      dw = grads[p] # 取出模型参数p对应的梯度值
      config = self.optim_configs[p] # 取出模型参数p对应的优化超参数
      next_w, next_config = self.update_rule(w, dw, config) # 优化算法
      self.model.params[p] = next_w # 新参数替换掉旧的
      self.optim_configs[p] = next_config # 新超参数替换掉旧的，如动量v

  def check_accuracy(self, X, y, num_samples=None, batch_size=100):
    """
    Check accuracy of the model on the provided data.
    #4# 定义我们的check_accuracy() 函数，其仅在train()函数中调用
    计算训练集的准确率
    Inputs:
    - X: Array of data, of shape (N, d_1, ..., d_k)
    - y: Array of labels, of shape (N,)
    - num_samples: If not None, subsample the data and only test the model
      on num_samples datapoints.
    - batch_size: Split X and y into batches of this size to avoid using too
      much memory.
      
    Returns:
    - acc: Scalar giving the fraction of instances that were correctly
      classified by the model.
    """
    
    # Maybe subsample the data
    N = X.shape[0]
    if num_samples is not None and N > num_samples: # 从总图片中选取num_samples个图片
      mask = np.random.choice(N, num_samples)
      N = num_samples
      X = X[mask]
      y = y[mask]

    # Compute predictions in batches
    num_batches = N / batch_size
    if N % batch_size != 0:
      num_batches += 1
    y_pred = []
    for i in xrange(num_batches):
      start = i * batch_size
      end = (i + 1) * batch_size
      scores = self.model.loss(X[start:end]) #########################
      y_pred.append(np.argmax(scores, axis=1))
    y_pred = np.hstack(y_pred)
    acc = np.mean(y_pred == y)

    return acc

  #step5# 定义我们最重要的函数  
  def train(self):
    """
    Run optimization to train the model.
    首先要确定下来总共要进行的迭代的次数num_iterations
    """
    num_train = self.X_train.shape[0] 
    iterations_per_epoch = max(num_train / self.batch_size, 1) # 每次迭代的次数
    num_iterations = self.num_epochs * iterations_per_epoch # 总迭代次数

    # 开始迭代循环
    for t in xrange(num_iterations):
      self._step()
      """
      上面完成了一次神经网络的迭代。此时，模型的参数已经更新过一次，
      并且在self.loss_history中添加了一个新的loss值
      """
      # Maybe print training loss
      if self.verbose and t % self.print_every == 0:
        print '(Iteration %d / %d) loss: %f' % (
               t + 1, num_iterations, self.loss_history[-1])

      # At the end of every epoch, increment the epoch counter and decay the
      # learning rate.
      epoch_end = (t + 1) % iterations_per_epoch == 0
      if epoch_end: # 只有当t = iterations_per_epoch - 1 时为True
        self.epoch += 1 # 第一遍之后开始， 从0自加1为每遍计数
        for k in self.optim_configs: # 第一遍之后开始，每遍给学习率自乘一个衰减率
          self.optim_configs[k]['learning_rate'] *= self.lr_decay

      # Check train and val accuracy on the first iteration, the last
      # iteration, and at the end of each epoch.
      first_it = (t == 0) # 起始的t
      last_it = (t == num_iterations + 1) # 最后的t
      if first_it or last_it or epoch_end: # 在最开始/最后/每遍结束时
        train_acc = self.check_accuracy(self.X_train, self.y_train,
                                        num_samples=1000) # 随机取1000个训练图看准确率
        val_acc = self.check_accuracy(self.X_val, self.y_val) # 计算全部验证图片的准确率
        self.train_acc_history.append(train_acc)
        self.val_acc_history.append(val_acc)

        if self.verbose:# 在最开始/最后/每遍结束时，打印准确率等信息
          print '(Epoch %d / %d) train acc: %f; val_acc: %f' % (
                 self.epoch, self.num_epochs, train_acc, val_acc)

        # Keep track of the best model 
        if val_acc > self.best_val_acc:# 在最开始/最后/每遍结束时，比较当前验证集的准确率和过往最佳验证集
          self.best_val_acc = val_acc
          self.best_params = {}
          for k, v in self.model.params.iteritems():
            self.best_params[k] = v.copy() # copy()仅复制值过来

    # At the end of training swap the best params into the model
    self.model.params = self.best_params # 最后把得到的最佳模型参数存入到模型中


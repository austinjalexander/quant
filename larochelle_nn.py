from module_imports import *

def NN_SGD(source, binarize, gt, lt, vol, balance_labeled_data, scaler, X_train, y_train, X_validation, y_validation, h1, h2, epochs, Lambda, Reg, alpha, plot=False):
  '''
  x = np.array([[0],
                [1], 
                [2]])

  y = np.array([[0],
                [1]])
  '''

  features = X_train[0].shape[0]
  #h1 = 100
  #h2 = 100
  outputs = y_train[0].shape[0]

  w1_init = np.sqrt(6)/np.sqrt(h1+features)
  W1 = np.random.uniform(low=-w1_init, high=w1_init, size=(h1*features)).reshape(h1,features)
  b1 = np.zeros((h1,1))
  #print "W1", W1.shape
  #print "b1", b1.shape

  w2_init = np.sqrt(6)/np.sqrt(h2+h1)
  W2 = np.random.uniform(low=-w2_init, high=w2_init, size=(h1*h2)).reshape(h1,h2)
  b2 = np.zeros((h2,1))
  #print "W2", W2.shape
  #print "b2", b2.shape

  w3_init = np.sqrt(6)/np.sqrt(outputs+h2)
  W3 = np.random.uniform(low=-w3_init, high=w3_init, size=(outputs*h2)).reshape(outputs,h2)
  b3 = np.zeros((outputs,1))
  #print "W3", W3.shape
  #print "b3", b3.shape

  #f_x = []

  #print "-"*10

  def loss(f_x,y):
    i = np.where(y == 1)[0][0]
    return -np.log(f_x[i]) # negative log-likelihood
    #print f_x, y
    #print
    #print -np.log(f_x)
    #print
    #return -np.log(f_x)
    #print f_x[1], y[1]
    #return -np.log(f_x[1])

  def forward_prop(x, W1, b1, W2, b2, W3, b3):
    def sigm(z):
      return 1/(1+np.exp(-z))

    def softmax(z):
      return np.exp(z)/np.sum(np.exp(z))

    z1 = b1 + np.dot(W1,x)
    a1 = sigm(z1)

    z2 = b2 + np.dot(W2,a1)
    a2 = sigm(z2)

    z3 = b3 + np.dot(W3,a2)
    a3 = softmax(z3)

    f_x = a3
    return z1, a1, z2, a2, z3, a3, f_x
  '''
  z1, a1, z2, a2, z3, a3, f_x = forward_prop(x, W1, b1, W2, b2, W3, b3)
  print "y\n", y
  print "f_x\n", f_x
  print "loss(f_x,y)\n", loss(f_x,y)
  print "-"*10
  '''

  def back_prop(x, W1, b1, W2, b2, W3, b3, z1, a1, z2, a2, z3, a3, f_x, y):

    def sigm(z):
      return 1/(1+np.exp(-z))

    def sigm_prime(z):
      return (sigm(z) * (1 - sigm(z)))

    del_z3 = -(y - f_x)
    del_W3 = np.dot(del_z3,a2.T)
    del_b3 = del_z3

    del_a2 = np.dot(W3.T,del_z3)
    del_z2 = np.multiply(del_a2,sigm_prime(z2))
    del_W2 = np.dot(del_z2,a1.T)
    del_b2 = del_z2

    del_a1 = np.dot(W2.T,del_z2)
    del_z1 = np.multiply(del_a1,sigm_prime(z1))
    del_W1 = np.dot(del_z1,x.T)
    del_b1 = del_z1

    return del_W1, del_b1, del_W2, del_b2, del_W3, del_b3
  #del_W1, del_b1, del_W2, del_b2, del_W3, del_b3 = back_prop(x, W1, b1, W2, b2, W3, b3, z1, a1, z2, a2, z3, a3, f_x, y)

  def finite_diff_approx(W, b, del_W, del_b, x, f_x, y):

    epsilon = 1e-6

    # W
    approx_del_W = []
    for i in xrange(W.shape[0]):
      for j in xrange(W.shape[1]):

        temp_w = (W[i][j])
        W[i][j] = W[i][j]+epsilon
        z1, a1, z2, a2, z3, a3, f_x = forward_prop(x, W, b1, W2, b2, W3, b3)
        loss_left = loss(f_x,y)[0]
        W[i][j] = temp_w

        W[i][j] = (W[i][j]-epsilon)
        z1, a1, z2, a2, z3, a3, f_x = forward_prop(x, W, b1, W2, b2, W3, b3)
        loss_right = loss(f_x,y)[0]
        W[i][j] = temp_w

        approx_del_W.append((loss_left - loss_right)/(2*epsilon))

    print "\nW gradient checking:"
    print "\tapprox_del_W\n\t",approx_del_W[:3]
    print "\tdel_W\n\t",del_W.ravel()[:3]
    print "\tapprox absolute difference:", np.sum(np.abs(approx_del_W - del_W.ravel()))/(len(approx_del_W)**2)

    # b
    approx_del_b = []
    for i in xrange(b.shape[0]):
      for j in xrange(b.shape[1]):

        temp_b = b[i][j]
        b[i][j] = (b[i][j]+epsilon)
        z1, a1, z2, a2, z3, a3, f_x = forward_prop(x, W1, b, W2, b2, W3, b3)
        loss_left = loss(f_x,y)[0]
        b[i][j] = temp_b

        b[i][j] = (b[i][j]-epsilon)
        z1, a1, z2, a2, z3, a3, f_x = forward_prop(x, W1, b, W2, b2, W3, b3)
        loss_right = loss(f_x,y)[0]
        b[i][j] = temp_b

        approx_del_b.append((loss_left - loss_right)/(2*epsilon))

    print "\nb gradient checking:"
    print "\tapprox_del_b\n\t",approx_del_b[:3]
    print "\tdel_b\n\t",del_b.ravel()[:3]
    print "\tapprox absolute difference:", np.sum(np.abs(approx_del_b - del_b.ravel()))/(len(approx_del_b)**2)

  #finite_diff_approx(W1, b1, del_W1, del_b1, x, f_x, y)

  def regularizer(Reg, W):
    if Reg == 'L2':
      # np.linalg.norm(W1)**2 + np.linalg.norm(W2)**2 + np.linalg.norm(W3)**2 # L2 regularization
      return (2 * W) # W L2 gradient
    elif Reg == 'L1':
      # np.sum(np.abs(W1)) + np.sum(np.abs(W2)) + np.sum(np.abs(W3)) # L1 regularization
      return np.sign(W) # W L1 gradient


  # SGD
  #delta = 0.7 # 0.5 < delta <= 1
  training_losses = []
  validation_losses = []
  mean_training_losses = []
  mean_validation_losses = []
  time0 = time()
  best_last_epoch = -np.inf
  best_last_mean_validation_loss = np.inf
  W = []
  b = []

  for i in xrange(epochs):

    indices = np.random.randint(low=0,high=X_train.shape[0],size=X_train.shape[0])
    X_train = X_train[indices]
    y_train = [y_train[index] for index in indices]

    # training
    for x,y in zip(X_train, y_train):

      x = x.reshape(x.shape[0],1)
      y = y.reshape(y.shape[0],1)

      z1, a1, z2, a2, z3, a3, f_x = forward_prop(x, W1, b1, W2, b2, W3, b3)
      del_W1, del_b1, del_W2, del_b2, del_W3, del_b3 = back_prop(x, W1, b1, W2, b2, W3, b3, z1, a1, z2, a2, z3, a3, f_x, y)

      deriv_W3 = -del_W3 - (Lambda * regularizer(Reg, W3))
      deriv_b3 = -del_b3
      W3 = W3 + (alpha * deriv_W3)
      b3 = b3 + (alpha * deriv_b3)

      deriv_W2 = -del_W2 - (Lambda * regularizer(Reg, W2))
      deriv_b2 = -del_b2
      W2 = W2 + (alpha * deriv_W2)
      b2 = b2 + (alpha * deriv_b2)    

      deriv_W1 = -del_W1 - (Lambda * regularizer(Reg, W1))
      deriv_b1 = -del_b1
      W1 = W1 + (alpha * deriv_W1)
      b1 = b1 + (alpha * deriv_b1)

      if np.isnan(W1[0])[0] == True:
          raise ValueError('A very specific bad thing happened')

      W = [W1, W2, W3]
      b = [b1, b2, b3]

      training_loss = np.round(loss(f_x,y),2)
      training_losses.append(training_loss)

    # validation
    for x,y in zip(X_validation, y_validation):

      x = scaler.transform(x)
      x = x.reshape(x.shape[0],1)
      y = y.reshape(y.shape[0],1)

      z1, a1, z2, a2, z3, a3, f_x = forward_prop(x, W1, b1, W2, b2, W3, b3)

      if np.isnan(W1[0])[0] == True:
          raise ValueError('A very specific bad thing happened')

      validation_loss = np.round(loss(f_x,y),2)
      validation_losses.append(validation_loss)

    # post-train, post-validation track losses
    
    mean_training_losses.append(np.mean(training_losses))
    mean_validation_losses.append(np.mean(validation_losses))
    
    last_mean_training_loss = np.round(mean_training_losses[-1],2)
    last_mean_validation_loss = np.round(mean_validation_losses[-1],2)

    if len(mean_validation_losses) > 1:
      last_mean_validation_loss_slope = (mean_validation_losses[-1] - mean_validation_losses[-2])/1.0

      if (last_mean_validation_loss <= best_last_mean_validation_loss) and (last_mean_validation_loss_slope <= 0):
          best_last_epoch = i
          best_last_mean_validation_loss = last_mean_validation_loss

    #if i > 5:
    #    alpha = alpha/(1+(delta*i))

    if (i != 0) and (i % 20 == 0) and (plot == True):        
      print "h1:",h1,"h2:",h2,"epochs:",epochs,"Lambda:",Lambda,"Reg:",Reg,"alpha:",alpha 
      print "last mean validation loss:", last_mean_validation_loss
      print "last mean training loss:", last_mean_training_loss
      plt.title('EPOCH ' + str(i))
      plt.plot(mean_validation_losses)
      plt.plot(mean_training_losses)
      plt.legend(['Validation', 'Training'])
      plt.xlabel('Epochs')
      plt.ylabel('Loss')
      plt.show()

  # reporting
  experiment_time = np.round((time()-time0), 2)
  
  last_mean_training_loss = np.round(mean_training_losses[-1],2)
  last_mean_validation_loss = np.round(mean_validation_losses[-1],2)
  
  last_mean_training_loss_slope = (mean_training_losses[-1] - mean_training_losses[-2])/1.0
  last_mean_validation_loss_slope = (mean_validation_losses[-1] - mean_validation_losses[-2])/1.0
  
  nn_report_df = pd.read_csv('nn_report.csv')
  data_to_record = [source,binarize,gt,lt,vol,balance_labeled_data,X_train.shape[0],features,outputs,h1,h2,epochs,Lambda,Reg,alpha,experiment_time,last_mean_training_loss,last_mean_validation_loss,last_mean_training_loss_slope,last_mean_validation_loss_slope,best_last_epoch,best_last_mean_validation_loss]
  data_to_record = np.array(data_to_record).reshape(1,len(data_to_record))
  data_df = pd.DataFrame(data_to_record, columns=nn_report_df.columns)

  nn_report_df = nn_report_df.append(data_df)
  nn_report_df.to_csv('nn_report.csv', index=False)

  #if plot == True:        
  #    print "h1:",h1,"h2:",h2,"epochs:",epochs,"Lambda:",Lambda,"Reg:",Reg,"alpha:",alpha 
  #    print "last mean validation loss:", last_mean_validation_loss
  #    print "last mean training loss:", last_mean_training_loss
  #    plt.title('EPOCH ' + str(i))
  #    plt.plot(mean_validation_losses)
  #    plt.plot(mean_training_losses)
  #    plt.legend(['Validation', 'Training'])
  #    plt.xlabel('Epochs')
  #    plt.ylabel('Loss')
  #    plt.show()

  return W,b

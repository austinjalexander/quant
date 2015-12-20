import os
from time import time
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest

from sklearn.svm import SVR
from sklearn.svm import SVC

def NN(cols,X,Y):

  # BALANCE
  balance_labeled_data = False
  # BALANCE LABELS
  if balance_labeled_data == True:
    # randomly balance labeled data
    indices_Y_is_0 = np.where(Y == 0)[0]
    #print indices_Y_is_0.shape[0]
    indices_Y_is_1 = np.where(Y == 1)[0]
    #print indices_Y_is_1.shape[0]

    subset_indices_Y_is_0 = np.random.choice(indices_Y_is_0, indices_Y_is_1.shape[0])
    X_is_0 = X[subset_indices_Y_is_0]
    Y_is_0 = Y[subset_indices_Y_is_0]
    X_is_1 = X[indices_Y_is_1]
    Y_is_1 = Y[indices_Y_is_1]

    X = np.concatenate((X_is_0,X_is_1))
    Y = np.concatenate((Y_is_0,Y_is_1))
    
    plt.hist(Y)
    plt.show()

  # SELECTKBEST
  k = np.random.randint(1,X.shape[1]+1)
  skb = SelectKBest(k=k)
  skb.fit(X,Y)
  for i in xrange(cols.shape[0]):
    if skb.scores_[i] in np.sort(skb.scores_)[-k:]:
      print cols[i], skb.scores_[i]
          
  skb_X = skb.transform(X)
  print "-"*10

  # VECTORIZE LABEL
  vectorize_label = True
  vect_Y = []
  # VECTORIZE LABELS
  if vectorize_label == True:
    for i in xrange(Y.shape[0]):
      if Y[i] == 0:
        vect_Y.append(np.array([[1],[0]]))
      elif Y[i] == 1:
        vect_Y.append(np.array([[0],[1]]))
  #print type(vect_Y)
  #print vect_Y[0]

  # BREAK UP DATA
  X_train, X_val_test, y_train, y_val_test = train_test_split(skb_X, vect_Y, test_size=0.3, random_state=42)
  X_validation, X_test, y_validation, y_test = train_test_split(X_val_test, y_val_test, test_size=0.5, random_state=42)

  # SCALE
  scaler = StandardScaler()
  scaler.fit(X_train)
  X_train = scaler.transform(X_train)

  #print X_train.shape
  #print len(y_train)

  # NETWORK FUNCTIONS
  def loss(f_x,y):
    i = np.where(y == 1)[0][0]
    return -np.log(f_x[i]) # negative log-likelihood

  def forward_prop(x, W1, b1, W2, b2, W3, b3):
    def sigm(z):
      return 1/(1+np.exp(-z))

    def tanh(z):
      return (np.exp(2*z)-1)/(np.exp(2*z)+1)

    def lecun_sig(z):
      return 1.7159 * tanh((2.0/3.0)*z)

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

  def back_prop(x, W1, b1, W2, b2, W3, b3, z1, a1, z2, a2, z3, a3, f_x, y):

    def sigm(z):
      return 1/(1+np.exp(-z))

    def sigm_prime(z):
      return (sigm(z) * (1 - sigm(z)))

    def tanh(z):
      return (np.exp(2*z)-1)/(np.exp(2*z)+1)

    def tanh_prime(z):
      return (1 - (tanh(z)**2))

    def lecun_sig(z):
      return 1.7159 * tanh((2.0/3.0)*z)

    def lecun_sig_prime(z):
      numerator = (2 * np.exp((2.0/3.0) * -z))
      denominator = (1 + np.exp(-2 * (2.0/3.0) * -z))
      return 1.14393 * (numerator/denominator)**2

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

  #x = X_train[0].reshape(X_train.shape[1],1)
  #y = y_train[0]
  #z1, a1, z2, a2, z3, a3, f_x = forward_prop(x, W1, b1, W2, b2, W3, b3)

  #del_W1, del_b1, del_W2, del_b2, del_W3, del_b3 = back_prop(x, W1, b1, W2, b2, W3, b3, z1, a1, z2, a2, z3, a3, f_x, y)
  #finite_diff_approx(W1, b1, del_W1, del_b1, x, f_x, y)

  def regularizer(Reg, W):
    if Reg == 'L2':
      # np.linalg.norm(W1)**2 + np.linalg.norm(W2)**2 + np.linalg.norm(W3)**2 # L2 regularization
      return (2 * W) # W L2 gradient
    elif Reg == 'L1':
      # np.sum(np.abs(W1)) + np.sum(np.abs(W2)) + np.sum(np.abs(W3)) # L1 regularization
      return np.sign(W) # W L1 gradient

  # STRUCTURE
  features = X_train[0].shape[0]
  print "features", features
  outputs = y_train[0].shape[0]
  print "outputs", outputs

  h1 = np.random.randint(1,features+100)
  h2 = np.random.randint(1,features+100)
  print "h1", h1
  print "h2", h2

  w1_init = np.sqrt(6)/np.sqrt(h1+features)
  W1 = np.random.uniform(low=-w1_init, high=w1_init, size=(h1*features)).reshape(h1,features)
  b1 = np.zeros((h1,1))
  print "W1", W1.shape
  print "b1", b1.shape

  w2_init = np.sqrt(6)/np.sqrt(h2+h1)
  W2 = np.random.uniform(low=-w2_init, high=w2_init, size=(h2*h1)).reshape(h2,h1)
  b2 = np.zeros((h2,1))
  print "W2", W2.shape
  print "b2", b2.shape

  w3_init = np.sqrt(6)/np.sqrt(outputs+h2)
  W3 = np.random.uniform(low=-w3_init, high=w3_init, size=(outputs*h2)).reshape(outputs,h2)
  b3 = np.zeros((outputs,1))
  print "W3", W3.shape
  print "b3", b3.shape

  print "-"*10

  # HYPERPARAMETERS
  alpha = np.random.choice([0.0001, 0.001, 0.01, 0.1])
  delta = np.random.choice([0.6, 0.7, 0.8, 0.9, 1.0]) # 0.5 < delta <= 1
  Lambda = np.random.choice([0.0001, 0.001, 0.01, 0.1, 1.0])
  Reg = np.random.choice(['L1', 'L2'])

  # SGD
  epochs = 50

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

    # shuffle examples each epoch
    indices = np.random.randint(low=0,high=X_train.shape[0],size=X_train.shape[0])
    X_train = X_train[indices]
    y_train = [y_train[index] for index in indices]

    # keep track of training examples;
    # only let positive label example go through first
    last = np.array([[1],
                     [0]])

    # training
    for x,y in zip(X_train, y_train):
      x = x.reshape(x.shape[0],1)  

      # skip example if same label as last iteration
      if np.array_equal(last, y):
        continue
      else:
        last = y

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
      W1 = W1 + ((alpha*5) * deriv_W1) # Lecun: learning rate larger
      b1 = b1 + ((alpha*5) * deriv_b1) # in lower layers

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

    # decrease alpha
    if i > 20:
      alpha = alpha/(1+(delta*i))

    if (i != 0) and (i % 20 == 0):        
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
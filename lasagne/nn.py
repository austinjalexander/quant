from __future__ import print_function

import sys
import os
import time

import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
import theano
import theano.tensor as T

import lasagne


# ################## Download and prepare the MNIST dataset ##################
# This is just some way of getting the MNIST dataset from an online location
# and loading it into numpy arrays. It doesn't involve Lasagne at all.

def load_dataset():
  def modify_columns(ticker, normalize):
    df = pd.read_csv("/Users/excalibur/Dropbox/datasets/quandl_data/{}.csv".format(ticker))
    df = df.drop('Adjusted Close', axis=1)
    
    df['50dravg'] = pd.rolling_mean(df['Close'], window=50)
    df['200dravg'] = pd.rolling_mean(df['Close'], window=200)

    df['50dravg'] = pd.rolling_mean(df['Close'], window=50)
    df['200dravg'] = pd.rolling_mean(df['Close'], window=200)

    if normalize == True:
      temp_df = df['Volume']
      df = df.drop('Volume', axis=1)
      std_df = df.std(axis=1, ddof=0)
      
      df['mean'] = df.mean(axis=1)
      df['std'] = std_df

      df['Open'] = (df['Open'] - df['mean']) / df['std']
      df['High'] = (df['High'] - df['mean']) / df['std']
      df['Low'] = (df['Low'] - df['mean']) / df['std']
      df['Close'] = (df['Close'] - df['mean']) / df['std']
      
      df['50dravg'] = (df['50dravg'] - df['mean']) / df['std']
      df['200dravg'] = (df['200dravg'] - df['mean']) / df['std']

      df = df.drop(['mean', 'std'], axis=1)

      df['Volume'] = temp_df

    df['OC%'] = (df['Close'] / df['Open']) - 1
    df['HL%'] = (df['High'] / df['Low']) - 1
    
    df['ticker'] = ticker

    df['label'] = df['OC%'].shift(-1)
    
    return df #df.loc[500:] # remove first 500 days


  def get_quandl_data(binarize=False, gt=2.0, lt=10.0, vol=10**5):
      
    tickers = [filename[:-4] for filename in os.listdir('/Users/excalibur/Dropbox/datasets/quandl_data/') if filename != '.DS_Store']

    normalize = False

    scale_volume = False

    # gather data
    stock_df = pd.DataFrame()
    for ticker in tickers:
      if stock_df.empty:
        stock_df = modify_columns(ticker, normalize)
      else:
        stock_df = stock_df.append(modify_columns(ticker, normalize))
        #stock_df = pd.concat([stock_df, modify_columns(ticker, normalize)])
        #stock_df = pd.concat([stock_df, modify_columns(ticker, normalize)], verify_integrity=True)
            
    # scale volume
    if scale_volume == True:     
      stock_df['Volume'] = (stock_df['Volume'] - stock_df['Volume'].min()) / (stock_df['Volume'].max() - stock_df['Volume'].min())
        
        # log volume
        #stock_df['Volume'] = stock_df['Volume'].map(lambda x: np.log(x))

    #stock_df = stock_df.drop(['Open', 'High', 'Low', 'Close'], axis=1)

    # add bias
    #stock_df.insert(0, 'bias', 1.0)

    # keep tickers for predictions
    pred_tickers = stock_df['ticker'].unique()

    # categoricalize tickers
    stock_df['ticker'] = stock_df['ticker'].astype('category').cat.codes

    # replace Infs with NaNs
    stock_df = stock_df.replace([np.inf, -np.inf], np.nan)

    # keep PPS > gt
    stock_df = stock_df[stock_df['Open'] > gt]

    # keep PPS < lt
    stock_df = stock_df[stock_df['Open'] < lt]

    # keep volume > vol
    stock_df = stock_df[stock_df['Volume'] > vol]

    prediction_df = stock_df.copy()

    #stock_df = stock_df.drop('ticker', axis=1)

    stock_df = stock_df.dropna()

    # binarize labels
    if binarize == True:
      stock_df['label'] = stock_df['label'].map(lambda x: 1 if x >= 0.05 else 0)

    return stock_df, prediction_df, pred_tickers
  #######################################################
  stock_df, prediction_df = pd.DataFrame(), pd.DataFrame()
  pred_tickers = []
  source = "Q"
  binarize = True
  gt = 0
  lt = 50.0
  vol = 0
  if source == "Q":
      stock_df, prediction_df, pred_tickers = get_quandl_data(binarize=binarize, gt=gt, lt=lt, vol=vol)
  elif source == "G":
      stock_df, prediction_df = get_goog_data(binarize=binarize, gt=gt, lt=lt, vol=vol)
      
  Y = stock_df['label'].values
  Y = Y.reshape(Y.shape[0], 1)

  X_df = stock_df.drop('label', axis=1)
  X = X_df.values

  #print X.shape, Y.shape
  #X_df.tail()

  vectorize_label = False
  if vectorize_label == True:
      new_y = []
      positives = []
      for i in xrange(Y.shape[0]):
          if Y[i] == 0:
              new_y.append(np.array([[1],[0]]))
          elif Y[i] == 1:
              new_y.append(np.array([[0],[1]]))
      Y = new_y

  X_train, X_vt, y_train, y_vt = train_test_split(X, Y, test_size=0.30, random_state=42)
  X_val, X_test, y_val, y_test = train_test_split(X_vt, y_vt, test_size=0.50, random_state=42)

  scaler = StandardScaler()
  X_train = scaler.fit_transform(X_train)
  return X_train, y_train, X_val, y_val, X_test, y_test


# ##################### Build the neural network model #######################
# This script supports three types of models. For each one, we define a
# function that takes a Theano variable representing the input and returns
# the output layer of a neural network model build in Lasagne.

def build_mlp(input_var=None):
    # This creates an MLP of two hidden layers of 800 units each, followed by
    # a softmax output layer of 10 units. It applies 20% dropout to the input
    # data and 50% dropout to the hidden layers.

    # Input layer, specifying the expected input shape of the network
    # (unspecified batchsize, 1 channel, 28 rows and 28 columns) and
    # linking it to the given Theano variable `input_var`, if any:
    l_in = lasagne.layers.InputLayer(shape=(10, 1),
                                     input_var=input_var)

    # Apply 20% dropout to the input data:
    l_in_drop = lasagne.layers.DropoutLayer(l_in, p=0.2)

    # Add a fully-connected layer of 800 units, using the linear rectifier, and
    # initializing weights with Glorot's scheme (which is the default anyway):
    l_hid1 = lasagne.layers.DenseLayer(
            l_in_drop, num_units=800,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())

    # We'll now add dropout of 50%:
    l_hid1_drop = lasagne.layers.DropoutLayer(l_hid1, p=0.5)

    # Another 800-unit layer:
    l_hid2 = lasagne.layers.DenseLayer(
            l_hid1_drop, num_units=800,
            nonlinearity=lasagne.nonlinearities.rectify)

    # 50% dropout again:
    l_hid2_drop = lasagne.layers.DropoutLayer(l_hid2, p=0.5)

    # Finally, we'll add the fully-connected output layer, of 10 softmax units:
    l_out = lasagne.layers.DenseLayer(
            l_hid2_drop, num_units=10,
            nonlinearity=lasagne.nonlinearities.softmax)

    # Each layer is linked to its incoming layer(s), so we only need to pass
    # the output layer to give access to a network in Lasagne:
    return l_out


def build_custom_mlp(input_var=None, depth=2, width=800, drop_input=.2,
                     drop_hidden=.5):
    # By default, this creates the same network as `build_mlp`, but it can be
    # customized with respect to the number and size of hidden layers. This
    # mostly showcases how creating a network in Python code can be a lot more
    # flexible than a configuration file. Note that to make the code easier,
    # all the layers are just called `network` -- there is no need to give them
    # different names if all we return is the last one we created anyway; we
    # just used different names above for clarity.

    # Input layer and dropout (with shortcut `dropout` for `DropoutLayer`):
    network = lasagne.layers.InputLayer(shape=(None, 1, 28, 28),
                                        input_var=input_var)
    if drop_input:
        network = lasagne.layers.dropout(network, p=drop_input)
    # Hidden layers and dropout:
    nonlin = lasagne.nonlinearities.rectify
    for _ in range(depth):
        network = lasagne.layers.DenseLayer(
                network, width, nonlinearity=nonlin)
        if drop_hidden:
            network = lasagne.layers.dropout(network, p=drop_hidden)
    # Output layer:
    softmax = lasagne.nonlinearities.softmax
    network = lasagne.layers.DenseLayer(network, 10, nonlinearity=softmax)
    return network


def build_cnn(input_var=None):
    # As a third model, we'll create a CNN of two convolution + pooling stages
    # and a fully-connected hidden layer in front of the output layer.

    # Input layer, as usual:
    network = lasagne.layers.InputLayer(shape=(None, 1, 28, 28),
                                        input_var=input_var)
    # This time we do not apply input dropout, as it tends to work less well
    # for convolutional layers.

    # Convolutional layer with 32 kernels of size 5x5. Strided and padded
    # convolutions are supported as well; see the docstring.
    network = lasagne.layers.Conv2DLayer(
            network, num_filters=32, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())
    # Expert note: Lasagne provides alternative convolutional layers that
    # override Theano's choice of which implementation to use; for details
    # please see http://lasagne.readthedocs.org/en/latest/user/tutorial.html.

    # Max-pooling layer of factor 2 in both dimensions:
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    # Another convolution with 32 5x5 kernels, and another 2x2 pooling:
    network = lasagne.layers.Conv2DLayer(
            network, num_filters=32, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify)
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    # A fully-connected layer of 256 units with 50% dropout on its inputs:
    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=.5),
            num_units=256,
            nonlinearity=lasagne.nonlinearities.rectify)

    # And, finally, the 10-unit output layer with 50% dropout on its inputs:
    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=.5),
            num_units=10,
            nonlinearity=lasagne.nonlinearities.softmax)

    return network


# ############################# Batch iterator ###############################
# This is just a simple helper function iterating over training data in
# mini-batches of a particular size, optionally in random order. It assumes
# data is available as numpy arrays. For big datasets, you could load numpy
# arrays as memory-mapped files (np.load(..., mmap_mode='r')), or write your
# own custom data iteration function. For small datasets, you can also copy
# them to GPU at once for slightly improved performance. This would involve
# several changes in the main program, though, and is not demonstrated here.

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]


# ############################## Main program ################################
# Everything else will be handled in our main program now. We could pull out
# more functions to better separate the code, but it wouldn't make it any
# easier to read.

def main(model='mlp', num_epochs=500):
    # Load the dataset
    print("Loading data...")
    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()

    # Prepare Theano variables for inputs and targets
    input_var = T.icol('inputs')
    target_var = T.icol('targets')

    # Create neural network model (depending on first command line parameter)
    print("Building model and compiling functions...")
    if model == 'mlp':
        network = build_mlp(input_var)
    elif model.startswith('custom_mlp:'):
        depth, width, drop_in, drop_hid = model.split(':', 1)[1].split(',')
        network = build_custom_mlp(input_var, int(depth), int(width),
                                   float(drop_in), float(drop_hid))
    elif model == 'cnn':
        network = build_cnn(input_var)
    else:
        print("Unrecognized model type %r." % model)

    # Create a loss expression for training, i.e., a scalar objective we want
    # to minimize (for our multi-class problem, it is the cross-entropy loss):
    prediction = lasagne.layers.get_output(network)
    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    loss = loss.mean()
    # We could add some weight decay as well here, see lasagne.regularization.

    # Create update expressions for training, i.e., how to modify the
    # parameters at each training step. Here, we'll use Stochastic Gradient
    # Descent (SGD) with Nesterov momentum, but Lasagne offers plenty more.
    params = lasagne.layers.get_all_params(network, trainable=True)
    updates = lasagne.updates.nesterov_momentum(
            loss, params, learning_rate=0.01, momentum=0.9)

    # Create a loss expression for validation/testing. The crucial difference
    # here is that we do a deterministic forward pass through the network,
    # disabling dropout layers.
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
                                                            target_var)
    test_loss = test_loss.mean()
    # As a bonus, also create an expression for the classification accuracy:
    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
                      dtype=theano.config.floatX)

    # Compile a function performing a training step on a mini-batch (by giving
    # the updates dictionary) and returning the corresponding training loss:
    train_fn = theano.function([input_var, target_var], loss, updates=updates, allow_input_downcast=True)

    # Compile a second function computing the validation loss and accuracy:
    val_fn = theano.function([input_var, target_var], [test_loss, test_acc])

    # Finally, launch the training loop.
    print("Starting training...")
    # We iterate over epochs:
    for epoch in range(num_epochs):
        # In each epoch, we do a full pass over the training data:
        train_err = 0
        train_batches = 0
        start_time = time.time()
        for batch in iterate_minibatches(X_train, y_train, 500, shuffle=True):
            inputs, targets = batch
            train_err += train_fn(inputs, targets)
            train_batches += 1

        # And a full pass over the validation data:
        val_err = 0
        val_acc = 0
        val_batches = 0
        for batch in iterate_minibatches(X_val, y_val, 500, shuffle=False):
            inputs, targets = batch
            err, acc = val_fn(inputs, targets)
            val_err += err
            val_acc += acc
            val_batches += 1

        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
        print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
        print("  validation accuracy:\t\t{:.2f} %".format(
            val_acc / val_batches * 100))

    # After training, we compute and print the test error:
    test_err = 0
    test_acc = 0
    test_batches = 0
    for batch in iterate_minibatches(X_test, y_test, 500, shuffle=False):
        inputs, targets = batch
        err, acc = val_fn(inputs, targets)
        test_err += err
        test_acc += acc
        test_batches += 1
    print("Final results:")
    print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
    print("  test accuracy:\t\t{:.2f} %".format(
        test_acc / test_batches * 100))

    # Optionally, you could now dump the network weights to a file like this:
    # np.savez('model.npz', lasagne.layers.get_all_param_values(network))


if __name__ == '__main__':
    if ('--help' in sys.argv) or ('-h' in sys.argv):
        print("Trains a neural network on MNIST using Lasagne.")
        print("Usage: %s [MODEL [EPOCHS]]" % sys.argv[0])
        print()
        print("MODEL: 'mlp' for a simple Multi-Layer Perceptron (MLP),")
        print("       'custom_mlp:DEPTH,WIDTH,DROP_IN,DROP_HID' for an MLP")
        print("       with DEPTH hidden layers of WIDTH units, DROP_IN")
        print("       input dropout and DROP_HID hidden dropout,")
        print("       'cnn' for a simple Convolutional Neural Network (CNN).")
        print("EPOCHS: number of training epochs to perform (default: 500)")
    else:
        kwargs = {}
        if len(sys.argv) > 1:
            kwargs['model'] = sys.argv[1]
        if len(sys.argv) > 2:
            kwargs['num_epochs'] = int(sys.argv[2])
        main(**kwargs)

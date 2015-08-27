import os
import requests
import csv
import re
import numpy as np
import pandas as pd
from scipy import stats
from scipy import optimize
import datetime
from time import time
import Quandl
#from matplotlib import pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
#import seaborn as sns

#from ticker_list import *

from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

from sklearn.feature_selection import SelectKBest
from sklearn.decomposition import RandomizedPCA 

from sklearn.preprocessing import PolynomialFeatures
 
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier 
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import BayesianRidge
from sklearn.linear_model import SGDRegressor 
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor 
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.grid_search import GridSearchCV

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
#######################################################
tickers = ['AAVL',
           'ABIO',
           'ACOR',
           'ADMA',
           'AERI',
           'AFFX',
           'AGEN',
           'APPY',
           'APTO',
           'ARDM',
           'ARIA',
           'ARNA',
           'ARWR',
           'ATRA',
           #'ATNM',  # AMEX
           #'AVXL',  # OTC
           #'AXN',   # AMEX
           'AXDX',
           'AXGN',
           'BABY',
           'BASI',
           'BCLI',
           'BCRX',
           'BGMD',
           'BIIB',
           'BLFS',
           'BLUE',
           'BOTA',
           'BRKR',
           'CAPN',
           'CASI',
           'CBLI',
           'CBMG',
           'CBMX',
           'CBPO',
           'CDTX',
           'CGEN',
           'CGNT',
           'CHRS',
           'CLDN',
           'CLDX',
           'CLLS',
           'CNMD',
           'COHR',
           'CPHD',
           'CPRX',
           'CRIS',
           'CUTR',
           'CYBX',
           'CYNO',
           'CYTR', 
           'DARA',
           'DBVT',
           'DRAD',
           'DSCO',
           'DYAX',
           'ECTE',
           'ECYT',
           'EDAP',
           'ELOS',
           'ENZN',
           'ESMC',
           'ETRM',
           'EXAS',
           'EXEL',
           'FATE',
           'FEIC',
           'FLDM',
           'FONR',
           'GENE',
           #'GEVA',  # not available on TD
           'GILD',
           'GNCA',
           'HALO',
           'HSKA',
           'IART',
           'ICCC',
           'IDRA',
           'IDXX',
           'ILMN',
           'IMMU',
           'IMRS',
           'INCR',
           'INCY',
           'INO',
           'IRIX',
           'JUNO',
           'KITE',
           'LOXO',
           'LPCN',
           'MEIP',
           'MNKD',
           'OREX',
           'PGNX',
           'QLTI',
           'RMTI',
           'SGYP',
           #'SNGX',  # OTC
           #'SPY',   # S&P 500
           #'SYN',   # AMEX
           'TENX',
           'THLD',
           'TNXP']
           #'TPIV']  # OTC
#######################################################
def check_quandl_latest(ticker):
  # check if last day's data is available
  print Quandl.get("YAHOO/{}".format(ticker), authtoken='DVhizWXNTePyzzy1eHWR').tail(1)

def download_quandl():

  start_tickers = tickers
  final_tickers = []

  print "\n", len(start_tickers), "total tickers to start\n"

  # download data
  for ticker in start_tickers:
      try:
          stock_df = Quandl.get("YAHOO/{}".format(ticker), authtoken='DVhizWXNTePyzzy1eHWR')
          stock_df.to_csv("quandl_data/{}.csv".format(ticker), index=False)
          final_tickers.append(ticker)
      except:
          print "removed:", ticker
              
  print "\n", len(final_tickers), "available tickers:"
  print final_tickers

#download_quandl()
#######################################################
def modify_columns(ticker, normalize):
    df = pd.read_csv("quandl_data/{}.csv".format(ticker))
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
    
    tickers = [filename[:-4] for filename in os.listdir('quandl_data/') if filename != '.DS_Store']

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

print X.shape, Y.shape
#X_df.tail()

indices_Y_is_0 = np.where(Y == 0)[0]
print indices_Y_is_0.shape[0]
indices_Y_is_1 = np.where(Y == 1)[0]
print indices_Y_is_1.shape[0]

subset_indices_Y_is_0 = np.random.choice(indices_Y_is_0, indices_Y_is_1.shape[0])
X_is_0 = X[subset_indices_Y_is_0]
Y_is_0 = Y[subset_indices_Y_is_0]
X_is_1 = X[indices_Y_is_1]
Y_is_1 = Y[indices_Y_is_1]

X = np.concatenate((X_is_0,X_is_1))
Y = np.concatenate((Y_is_0,Y_is_1))

vectorize_label = True
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
X_validation, X_test, y_validation, y_test = train_test_split(X_vt, y_vt, test_size=0.50, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
#######################################################
#######################################################
#######################################################
def NN_SGD(X_train, y_train, X_validation, y_validation, h1, h2, epochs, Lambda, Reg, alpha):
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

    print "-"*10

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
    best_epoch_mean_validation_loss = (0,1e10)

    for i in xrange(epochs):

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

        mean_training_losses.append(np.mean(training_losses))
        mean_validation_losses.append(np.mean(validation_losses))

        current_mean_validation_loss = np.round(mean_validation_losses[-1],2)
        current_mean_training_loss = np.round(mean_training_losses[-1],2)

        if current_mean_validation_loss < best_epoch_mean_validation_loss[1]:
            best_epoch_mean_validation_loss = (i,current_mean_validation_loss)

        #if (i != 0):        
        #    print "h1:",h1,"h2:",h2,"epochs:",epochs,"Lambda:",Lambda,"Reg:",Reg,"alpha:",alpha 
        #    print "current mean validation loss:", current_mean_validation_loss
        #    print "current mean training loss:", current_mean_training_loss
        #    plt.title('EPOCH ' + str(i))
        #    plt.plot(mean_validation_losses)
        #    plt.plot(mean_training_losses)
        #    plt.legend(['Validation', 'Training'])
        #    plt.xlabel('Epochs')
        #    plt.ylabel('Loss')
        #    plt.show()

        #if i > 5:
        #    alpha = alpha/(1+(delta*i))

    experiment_time = np.round((time()-time0), 2)
    nn_report_df = pd.read_csv('nn_report.csv')
    data_to_record = [source, binarize, gt, lt, vol, features, outputs, h1, h2, epochs, Lambda, Reg, alpha, experiment_time, np.round(mean_training_losses[-1],2), np.round(mean_validation_losses[-1],2), best_epoch_mean_validation_loss[0], best_epoch_mean_validation_loss[1]]
    data_to_record = np.array(data_to_record).reshape(1,len(data_to_record))
    data_df = pd.DataFrame(data_to_record, columns=nn_report_df.columns)

    nn_report_df = nn_report_df.append(data_df)
    nn_report_df.to_csv('nn_report.csv', index=False)
#######################################################
h1s = [10, 50, 100, 500]
h2s = [10, 50, 100, 500]

epochs = 500 
Lambdas = [1.0, 0.1, 0.01, 0.001]
Regs = ['L1', 'L2']
alphas = [1, 0.1, 0.01, 0.001]

# GRID SEARCH
for h1 in h1s:
    for h2 in h2s:
        for Lambda in Lambdas:
            for Reg in Regs:
                for alpha in alphas:
                    NN_SGD(X_train, y_train, X_validation, y_validation, h1, h2, epochs, Lambda, Reg, alpha)
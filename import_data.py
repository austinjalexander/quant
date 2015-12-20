from module_imports import *

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
    #df['label'] = df['HL%'].shift(-1)
    
    return df #df.loc[500:] # remove first 500 days


def get_quandl_data():
    
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
    #stock_df['ticker'] = stock_df['ticker'].astype('category').cat.codes

    # replace Infs with NaNs
    stock_df = stock_df.replace([np.inf, -np.inf], np.nan)

    prediction_df = stock_df.copy()

    #stock_df = stock_df.drop('ticker', axis=1)

    stock_df = stock_df.dropna()

    return stock_df, prediction_df, pred_tickers

#########################################

def flatten_goog_data(ticker):
    
  stock_df = pd.read_csv("/Users/excalibur/Dropbox/datasets/goog_data/{}.csv".format(ticker))

  # intraday drop last rows
  stock_df = stock_df.drop(stock_df.index[[-1,-2]])
  
  #print stock_df['time'].value_counts()
  stock_df = stock_df[(stock_df['time'] != 9.3) & (stock_df['time'] != 16.0)]
  #print stock_df['time'].value_counts()
  times = stock_df['time'].unique()
  
  columns = list(stock_df.columns)
  
  new_columns = []
  
  for time in times:
      for column in columns:
          new_columns.append(str(time) + "-" + column)
  
  #print "number of flattened columns", len(new_columns)
  #print new_columns
  
  flat_values = stock_df.values.ravel()
  stock_df = pd.DataFrame(columns=new_columns)
  
  errors = 0
  for day in xrange(len(flat_values)/len(new_columns)):
      
    day_values = flat_values[:len(new_columns)]
    
    #if day == 0:
    #    print list(day_values)

    if day_values[0] != 10.0 or day_values[-10] != 15.3:
        errors += 1
        continue
    else:
        df = pd.DataFrame([list(day_values)], columns=new_columns)
        stock_df = stock_df.append(df)
    
        flat_values = flat_values[len(new_columns):]

  #print "number of errors:", errors, "for", ticker

  stock_df['label'] = ((stock_df['15.3-CLOSE'] / stock_df['10.0-OPEN']) - 1).shift(-1)

  stock_df['ticker'] = ticker

  return stock_df

def get_goog_data():

  tickers = [filename[:-4] for filename in os.listdir('/Users/excalibur/Dropbox/datasets/goog_data/') if filename != '.DS_Store']

  # gather data
  stock_df = pd.DataFrame()
  for ticker in tickers:
    if stock_df.empty:
        stock_df = flatten_goog_data(ticker)
    else:
      df = flatten_goog_data(ticker)
      if not df.empty and len(df.columns) == len(stock_df.columns):
        stock_df = stock_df.append(df)

  week_days = stock_df['10.0-week_day'].values
  dates = stock_df['10.0-date'].values
  stock_df = stock_df.drop([col for col in stock_df.columns if ("time" in col) or ("week_day" in col) or ("date" in col)], axis=1)
  stock_df['week_day'] = week_days
  stock_df['date'] = dates

  # keep tickers for predictions
  pred_tickers = stock_df['ticker'].unique()

  # categoricalize tickers
  #stock_df['ticker'] = stock_df['ticker'].astype('category').cat.codes

  # replace Infs with NaNs
  stock_df = stock_df.replace([np.inf, -np.inf], np.nan)

  prediction_df = stock_df.copy()

  #stock_df = stock_df.drop('ticker', axis=1)

  stock_df = stock_df.dropna()

  return stock_df, prediction_df, pred_tickers



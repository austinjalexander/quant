from module_imports import *

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

      # keep dates
      dates = stock_df.index.values
      stock_df['date'] = dates

      stock_df.to_csv("/Users/excalibur/Dropbox/datasets/quandl_data/{}.csv".format(ticker), index=False)
      final_tickers.append(ticker)
    except:
      print "removed:", ticker
              
  print "\n", len(final_tickers), "available tickers:"
  print final_tickers

################################

def download_goog_stock(ticker, seconds_interval, num_of_days):
    
    url = "http://www.google.com/finance/getprices?q={0}&i={1}&p={2}d&f=d,o,h,l,c,v".format(ticker, seconds_interval, num_of_days)

    # get data and convert to data frame
    stock_df = pd.read_csv(url, skiprows=[0,1,2,3,5,6])

    # rename column name
    stock_df.rename(columns={'COLUMNS=DATE':'time'}, inplace=True)

    # remove 'a' from unix timestamps
    stock_df.replace(to_replace={'time':{'a':''}}, regex=True, inplace=True)

    # get entire column and convert to ints
    time_indices = [int(x) for x in stock_df['time'].values]

    # keep track of current timestamp
    last_timestamp = time_indices[0]

    # convert unix timestamp abbreviations into full unix timestamps
    for i in range(len(time_indices)):
        if time_indices[i] < last_timestamp:
            time_indices[i] = last_timestamp + (time_indices[i] * seconds_interval)
        else:
            last_timestamp = time_indices[i]

    # convert unix timestamps to human-readable formats
    time_indices = [datetime.datetime.fromtimestamp(x).strftime('%Y-%m-%d %H:%M:%S') for x in time_indices]

    # keep dates (i.e., not times)
    dates = [x for x in time_indices]
    dates = [date[:10] for date in dates]

    # create new column in data frame
    stock_df['date'] = dates
    
    # keep times (i.e., not dates)
    times = [float(x[-8:-3].replace(':','.')) for x in time_indices]

    # create new column in data frame
    stock_df['time'] = times

    # keep day of month
    #dates = [int(x[:10].split('-')[2]) for x in time_indices]
    # create new column in data frame
    #stock_df['month_date'] = dates

    # get weekday as int value
    stock_df['week_day'] = [datetime.datetime.strptime(x[:10], '%Y-%m-%d').weekday() for x in time_indices]

    # create features
    stock_df['op_cl%'] = np.true_divide(stock_df['CLOSE'], stock_df['OPEN']) - 1
    stock_df['lo_hi%'] = np.true_divide(stock_df['HIGH'], stock_df['LOW']) - 1
    #stock_df['vol_norm'] = np.true_divide(stock_df['VOLUME'], np.max(stock_df['VOLUME']))

    # create labels dataframe
    labels_df = stock_df.copy(deep=True)

    # remove columns
    #stock_df = stock_df.drop(['CLOSE', 'OPEN', 'LOW', 'HIGH', 'VOLUME'], axis=1)
    
    return stock_df

def download_goog():
  start_tickers = tickers

  for ticker in tickers:
      
    seconds_interval = 1800 # 1800: 30-minute (seems the most consistent)
    try:
      stock_df = download_goog_stock(ticker, seconds_interval, 1000)
        
      stock_df.to_csv("/Users/excalibur/Dropbox/datasets/goog_data/{}.csv".format(ticker), index=False)
    except: 
      print "problem with", ticker


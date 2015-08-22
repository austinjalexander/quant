from module_imports import *

def check_latest(ticker):
  # check if last day's data is available
  print Quandl.get("YAHOO/{}".format(ticker), authtoken='DVhizWXNTePyzzy1eHWR').tail(1)


def download():

  start_tickers = ticker_list.tickers
  tickers = []

  print "\n", len(start_tickers), "total tickers to start\n"

  # download data
  for ticker in start_tickers:
      try:
          stock_df = Quandl.get("YAHOO/{}".format(ticker), authtoken='DVhizWXNTePyzzy1eHWR')
          stock_df.to_csv("/Users/excalibur/Dropbox/datasets/quandl_data/{}.csv".format(ticker), index=False)
          tickers.append(ticker)
      except:
          print "removed:", ticker
              
  print "\n", len(tickers), "available tickers:"
  print tickers

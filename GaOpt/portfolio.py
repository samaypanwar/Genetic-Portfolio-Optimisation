
# OPTIMISE PORT BASED ON GA FOR ROLLING N MONTHS AND RETURN PORTFOLIO VALU
from pandas.tseries.offsets import QuarterEnd
from GaOpt.genetic import GeneticAlgorithm
from GaOpt.utils import sharpe, sortino, calmar, hybrid
import yfinance as yfin
import pandas as pd
from tqdm import tqdm
import datetime as dt


def backtest(tickers, verbose = False):

    start_date = '2020-09-01'
    dataframe = get_data(tickers, start_date)
    dates = pd.date_range(start=start_date, end=dt.date.today(), freq='Q').tolist()

    LOOKBACK = 2

    first = True
    backtest.weights = []

    for date in tqdm(dates[LOOKBACK:]):

        train = dataframe[date - QuarterEnd(LOOKBACK + 1): date - QuarterEnd(1)]

        test = dataframe[date - QuarterEnd(1): date]

        weights = GaOpt(train, verbose)
        print(weights)
        backtest.weights.append(weights)

        if first:

            portfolio = pd.DataFrame((test*weights).sum(axis=1) / (test*weights).sum(axis=1).iloc[0])
            portfolio.columns = ['Portfolio']
            first = False

        else:

            temp = pd.DataFrame((test*weights).sum(axis=1) / (test*weights).sum(axis=1).iloc[0])
            temp.columns = ['Portfolio']
            temp = temp * portfolio.iloc[-1, :]
            portfolio = pd.concat([portfolio, temp], axis=0)

    backtest.weights = pd.DataFrame(backtest.weights, index = dates[LOOKBACK:])
    backtest.weights.columns = dataframe.columns

    return portfolio

def get_data(tickers, start_date):

    dataframe = yfin.download(tickers, start=start_date, interval='1d')['Adj Close'].ffill().dropna()

    return dataframe

def GaOpt(train, verbose):

    GA = GeneticAlgorithm(hybrid, train, longOnly=True)
    bestWeights = GA.evolution(verbose)

    return bestWeights
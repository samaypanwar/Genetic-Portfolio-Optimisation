# DEF FITNESS FUNCTION HERE
from quant_risk.statistics import financial_ratios
from quant_risk.statistics import statistics
import numpy as np
import pandas as pd

def sharpe(dataframe, weights):

    portfolio = pd.DataFrame(np.dot(dataframe, weights), index=dataframe.index)
    statistic = financial_ratios.sharpe_ratio(portfolio)

    return statistic[0]

def calmar(dataframe, weights):

    portfolio = pd.DataFrame(np.dot(dataframe, weights), index=dataframe.index)
    statistic = financial_ratios.calmar_ratio(portfolio)

    return statistic[0]

def sortino(dataframe, weights):

    portfolio = pd.DataFrame(np.dot(dataframe, weights), index=dataframe.index)
    statistic = financial_ratios.sortino_ratio(portfolio)

    return statistic[0]

def hybrid(dataframe, weights):

    portfolio = pd.DataFrame(np.dot(dataframe, weights), index=dataframe.index)
    sortino_ratio = financial_ratios.sortino_ratio(portfolio)
    max_drawdown = statistics.maximum_drawdown(portfolio)

    statistic = 0.4 * sortino_ratio[0] - 0.6 * max_drawdown[0]

    return statistic
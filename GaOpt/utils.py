# DEF FITNESS FUNCTION HERE
from quant_risk.statistics import financial_ratios
import numpy as np

def sharpe(dataframe, weights):

    portfolio = np.dot(dataframe, weights)
    statistic = financial_ratios.sharpe_ratio(portfolio)

    return statistic

def calmar(dataframe, weights):

    portfolio = np.dot(dataframe, weights)
    statistic = financial_ratios.calmar_ratio(portfolio)

    return statistic

def sortino(dataframe, weights):

    portfolio = np.dot(dataframe, weights)
    statistic = financial_ratios.sortino_ratio(portfolio)

    return statistic

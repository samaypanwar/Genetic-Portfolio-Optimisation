
# OPTIMISE PORT BASED ON GA FOR ROLLING N MONTHS AND RETURN PORTFOLIO VALU
from genetic import GeneticAlgorithm
from utils import sharpe, sortino, calmar, hybrid
import yfinance as yfin

tickers = [
    'TSLA',
    'ETSY',
    'NVDA',
    'PYPL',
    'LB',
    'ALB',
    'FCX',
    'NOW',
    'ROL'
]

dataframe = yfin.download(tickers, period='1y')['Adj Close']

GA = GeneticAlgorithm(hybrid, dataframe, longOnly=True)
bestWeights = GA.evolution()

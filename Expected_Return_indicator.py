# -*- coding: utf-8 -*-
"""
Created on Sat Jan  1 19:48:28 2022

@author: VkVkV
"""

import numpy as np
import pandas as pd
import yfinance as yf
import datetime
import time

#return historical price
ticker = 'MDT'
data_stock = yf.Ticker(ticker).history(period='max')

#return historical SP500 price
data_market = yf.Ticker('SPY').history(period='max')

data_market

#Return historical 10y treasury note
rate = yf.Ticker('^TNX').history(period = 'max')

#make the 2 dataset to be the same length
if len(data_market) < len(data_stock):
    data_stock = data_stock.iloc[(len(data_stock) - len(data_market)):len(data_stock),:]
    rate = rate.iloc[(len(rate) - len(data_market)):len(rate),:]
    
elif len(data_market) >= len(data_stock):
    data_market = data_market.iloc[(len(data_market) - len(data_stock)):len(data_market), :]
    rate = rate.iloc[(len(rate) - len(data_stock)):len(rate),:]

def Daily_Return(Today_Price, Yesterday_Price):
        per_Return = 100*(Today_Price-Yesterday_Price)/Yesterday_Price
        return per_Return

data_market['Return'] = np.zeros(data_market.shape[0])
for i in range(1, len(data_market.index)):
    data_market["Return"][i] = Daily_Return(data_market["Close"][i],data_market["Close"][i-1])

data_stock['Return'] = np.zeros(data_stock.shape[0])
for i in range(1, len(data_stock.index)):
    data_stock["Return"][i] = Daily_Return(data_stock["Close"][i],data_stock["Close"][i-1])

data = pd.DataFrame(np.zeros(len(data_market)).astype(str), columns=['Date'])
data.loc[:,'Stock Return'] = 0.0
data.loc[:,'Market Return'] = 0.0
data.loc[:,'Beta'] = 0.0
data.loc[:,'Expected Stock Return'] = 0.0
data.loc[:,'Interest Rate'] = 0.0

data['Date'] = data_market.index
data['Stock Return'] = data_stock['Return'].to_numpy()
data['Market Return'] = data_market['Return'].to_numpy()
data.sort_index(ascending=False, inplace=True)

betas = []
for i in range(0, len(data_market)-500):
    # Calculate Covariance
    var_covar = np.cov(data["Stock Return"].values[i:500+i],data["Market Return"].values[i:500+i])
    covariance = var_covar[0,1]
    covariance

    # Calculate Variance
    variance = np.var(data["Stock Return"].values[i:500+i])
    variance

    #Calculate Beta
    beta = covariance/variance
    beta
    
    betas.append(beta)

data['Beta'][0:len(betas)] = betas

data = data.reset_index(drop=True)

for i in range(0, len(data)-500):
    if data['Date'][i] in rate.index:
        data['Interest Rate'][i] = rate.loc[rate.index == data['Date'][i], 'Close']
    elif data['Date'][i] not in rate.index:
        data['Interest Rate'][i] = rate.loc[rate.index == data['Date'][i-5], 'Close']

data_analysis = data.iloc[0:len(data)-500]
data_analysis['Interest Rate'] = data_analysis['Interest Rate']/100

#Calculate Expected Return using CAPM
data_analysis['Expected Stock Return'] = data_analysis['Interest Rate']+data_analysis['Beta']*(data_analysis['Market Return']+data_analysis['Interest Rate'])
data_analysis

data_analysis.sort_index(ascending=False, inplace=True)
data_analysis = data_analysis.reset_index(drop=True)
data_analysis

#Calculate different sma periods for the differences between stock Actual return and stock expected return
sma_20 = (data_analysis['Stock Return'] - data_analysis['Expected Stock Return']).rolling(window=20).mean()
sma_9 = (data_analysis['Stock Return'] - data_analysis['Expected Stock Return']).rolling(window=9).mean()

#Compare between Actual Return and Expected Return using CAPM
import matplotlib.pyplot as plt
import plotly.express as px
a = data_analysis['Date']
b = data_analysis['Stock Return']
c = data_analysis['Expected Stock Return']
fig = px.line(x = a, y = [b, c])
fig

if len(sma_20) >= 1000:
    sma_20_sort = np.sort(sma_20[(len(sma_20)-1000):len(sma_20)])
elif len(sma_20) < 1000:
    sma_20_sort = np.sort(sma_20)
    
max_overbought = np.mean(sma_20_sort[(len(sma_20_sort)-20):(len(sma_20_sort))])
max_oversold = np.mean(sma_20_sort[0:20])

import statistics as stats
in_pos = []
in_neg = []
for i in sma_20_sort:
    if i > 0:
        in_pos.append(i)
    if i < 0:
        in_neg.append(i)
in_pos_mean = np.mean(in_pos)
in_neg_mean = np.mean(in_neg)

overbought = stats.variance(in_pos) + in_pos_mean
oversold = in_neg_mean - stats.variance(in_neg) 

fig1 = (px.line(x = a, y = [sma_9, sma_20]))
# fig1.add_hline(y=0, line_color = 'white')
fig1.add_hline(y=overbought, line_color="green")
fig1.add_hline(y=oversold, line_color="red")
fig1.add_hline(y=max_overbought, line_color = 'chartreuse')
fig1.add_hline(y=max_oversold, line_color = 'brown')
fig1.add_hrect(y0=0, y1=max_overbought, 
              annotation_text="Buy Buy Buy", annotation_position="top left",
              fillcolor="green", opacity=0.25, line_width=0)
fig1.add_hrect(y0=0, y1=max_oversold, 
              annotation_text="Sell Sell Sell", annotation_position="bottom left",
              fillcolor="red", opacity=0.25, line_width=0)
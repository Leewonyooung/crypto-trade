"""Finite, trailing-only candle features. Maximum raw lookback is100 bars."""
import numpy as np
import pandas as pd
VERSION='ohlcv100-v1'
LOOKBACK=100

def build(frame):
    o,h,l,c,v=[frame[k].astype(float).reset_index(drop=True) for k in ['open','high','low','close','volume']]
    rng=(h-l).replace(0,np.nan);tr=pd.concat([h-l,(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1);atr=tr.rolling(20).mean().replace(0,np.nan)
    low=l.rolling(100).min();high=h.rolling(100).max();span=(high-low).replace(0,np.nan);X=pd.DataFrame(index=c.index)
    X['position100']=(c-low)/span;X['low_distance_atr']=(c-low)/atr;X['high_distance_atr']=(high-c)/atr
    X['lower_wick']=(pd.concat([o,c],axis=1).min(axis=1)-l)/rng;X['upper_wick']=(h-pd.concat([o,c],axis=1).max(axis=1))/rng
    X['body_fraction']=(c-o)/rng;X['close_location']=(c-l)/rng;X['range_atr']=rng/atr;X['gap_atr']=(o-c.shift())/atr
    X['range100_atr']=span/atr;X['range20_to100']=(h.rolling(20).max()-l.rolling(20).min())/span
    for lag in [1,2,3,5,10,20,40,60,80,99]:X[f'return_{lag}']=(c-c.shift(lag))/atr
    for size in [5,20,50]:
        sma=c.rolling(size).mean();X[f'sma{size}_distance']=(c-sma)/atr;X[f'sma{size}_slope5']=(sma-sma.shift(5))/atr
        X[f'bull_fraction{size}']=(c>o).astype(float).rolling(size).mean()
    delta=c.diff();up=delta.clip(lower=0).rolling(14).mean();down=(-delta).clip(lower=0).rolling(14).mean();rsi=100-100/(1+up/down);rsi[(up==0)&(down==0)]=50
    X['cutler_rsi14']=rsi;X['rsi_change3']=rsi.diff(3);X['stoch14']=(c-l.rolling(14).min())/(h.rolling(14).max()-l.rolling(14).min())
    X['atr5_to20']=tr.rolling(5).mean()/atr;X['atr20_to50']=atr/tr.rolling(50).mean()
    for size in [5,20,50]:X[f'volume_ratio{size}']=v/v.shift().rolling(size).mean()
    X['signed_volume20']=((2*c-h-l)/rng*v).rolling(20).sum()/v.rolling(20).sum()
    for lag in [1,2,3,5,10,20]:
        X[f'lower_wick_lag{lag}']=X.lower_wick.shift(lag);X[f'upper_wick_lag{lag}']=X.upper_wick.shift(lag)
    return X.replace([np.inf,-np.inf],np.nan)

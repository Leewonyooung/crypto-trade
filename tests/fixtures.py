"""Synthetic engineering fixtures. NEVER financial evidence or real human labels."""
import numpy as np,pandas as pd

def candles(n=6000):
    rng=np.random.default_rng(19);i=np.arange(n);close=50000+180*np.sin(i/28)+40*np.sin(i/7)+rng.normal(0,3,n);op=np.r_[close[0],close[:-1]]
    high=np.maximum(op,close)+rng.uniform(3,18,n);low=np.minimum(op,close)-rng.uniform(3,18,n)
    return pd.DataFrame(dict(timestamp=1735689600000+i*900000,open=op,high=high,low=low,close=close,volume=rng.uniform(10,200,n)))

def annotated(store,n=6000):
    f=candles(n);ds=store.import_frame(f,'SYNTHETIC TEST ONLY','TEST');rev=0
    for i in range(240,n-100,90):
        snap=store.label(ds['id'],int(f.timestamp.iloc[i]),'bottom' if (i//90)%2==0 else 'top',1,'synthetic fixture',rev);rev=snap['dataset']['revision']
    snap=store.review(ds['id'],int(f.timestamp.iloc[0]),int(f.timestamp.iloc[-1]),rev)
    return f,ds['id'],snap

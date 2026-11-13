from __future__ import annotations
import hashlib,io
import numpy as np
import pandas as pd

STEP=900_000

def parse_csv(content: bytes) -> pd.DataFrame:
    if len(content)>100*1024*1024:
        raise ValueError('CSV는 100 MiB 이하로 올려주세요.')
    try:
        df=pd.read_csv(io.BytesIO(content),float_precision='round_trip')
    except Exception as exc:
        raise ValueError('CSV를 읽을 수 없습니다.') from exc
    df.columns=[str(x).strip().lower() for x in df.columns]
    aliases={'open_time':'timestamp','time':'timestamp','datetime':'timestamp','date':'timestamp'}
    for source,target in aliases.items():
        if target not in df and source in df:df=df.rename(columns={source:target})
    required=['timestamp','open','high','low','close']
    if not set(required)<=set(df):raise ValueError('timestamp, open, high, low, close 컬럼이 필요합니다.')
    if not 120<=len(df)<=1_000_000:raise ValueError('120~1,000,000개의 완성된 15분봉을 사용하세요.')
    ts=pd.to_numeric(df.timestamp,errors='coerce')
    if ts.notna().all():
        if (ts%1!=0).any():raise ValueError('timestamp는 정수 epoch 또는 ISO 날짜여야 합니다.')
        if ts.median()<100_000_000_000:ts=ts*1000
        ts=ts.astype('int64')
    else:
        try:ts=pd.to_datetime(df.timestamp,utc=True,errors='raise').astype('int64')//1_000_000
        except Exception as exc:raise ValueError('timestamp를 UTC epoch/ISO 날짜로 해석할 수 없습니다.') from exc
    result=pd.DataFrame({'timestamp':ts})
    for col in required[1:]:result[col]=pd.to_numeric(df[col],errors='coerce')
    result['volume']=pd.to_numeric(df['volume'],errors='coerce') if 'volume' in df else np.nan
    result=result.sort_values('timestamp').reset_index(drop=True)
    if result.timestamp.duplicated().any():raise ValueError('중복 timestamp가 있습니다.')
    delta=np.diff(result.timestamp)
    if (delta<STEP).any() or (delta%STEP!=0).any():raise ValueError('15분봉 timestamp 간격이 아닙니다.')
    prices=result[['open','high','low','close']].to_numpy()
    if not np.isfinite(prices).all() or (prices<=0).any():raise ValueError('가격은 유한한 양수여야 합니다.')
    if (result.high<result[['open','close','low']].max(axis=1)).any() or (result.low>result[['open','close','high']].min(axis=1)).any():raise ValueError('OHLC 고가/저가 관계가 잘못됐습니다.')
    if ((result.volume<0)|np.isinf(result.volume)).any():raise ValueError('거래량은 0 이상 또는 결측이어야 합니다.')
    if (result.timestamp%STEP!=0).any():raise ValueError('timestamp는 15분봉 시작 시각이어야 합니다.')
    return result

def fingerprint(frame: pd.DataFrame) -> str:
    a=frame[['timestamp','open','high','low','close','volume']].to_numpy(dtype='<f8').copy()
    a[np.isnan(a)]=np.nan
    return hashlib.sha256(a.tobytes()).hexdigest()

def finite_records(frame: pd.DataFrame) -> list[dict]:
    return frame.replace([np.inf,-np.inf],np.nan).astype(object).where(frame.notna(),None).to_dict('records')

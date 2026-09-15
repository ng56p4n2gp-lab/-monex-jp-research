"""Validation gates for V13. Fail closed on temporal leakage/data defects."""
import pandas as pd

FUTURE_PREFIXES=('future_','mfe','mae','end')

def assert_chronological(train:pd.DataFrame,test:pd.DataFrame,time_col='decision_at'):
    a=pd.to_datetime(train[time_col]); b=pd.to_datetime(test[time_col])
    if len(a) and len(b) and a.max()>=b.min(): raise ValueError('train/test temporal overlap')
    return True

def assert_feature_names(features):
    bad=[x for x in features if x.lower().startswith(FUTURE_PREFIXES)]
    if bad: raise ValueError(f'future/evaluation columns used as features: {bad}')
    return True

def assert_ohlcv(df):
    req={'open','high','low','close','volume'}
    miss=req-set(df.columns)
    if miss: raise ValueError(f'missing OHLCV: {sorted(miss)}')
    bad=(df['high']<df[['open','close','low']].max(axis=1)) | (df['low']>df[['open','close','high']].min(axis=1)) | (df['volume']<0)
    if bad.any(): raise ValueError(f'invalid OHLCV rows: {int(bad.sum())}')
    return True

def assert_next_bar_entry(df,signal_col='decision_at',entry_col='entry_at'):
    s=pd.to_datetime(df[signal_col]); e=pd.to_datetime(df[entry_col])
    if (e<=s).any(): raise ValueError('entry is not strictly after decision time')
    return True

def official_known_before(decision_at,published_at):
    d=pd.to_datetime(decision_at,utc=True); p=pd.to_datetime(published_at,utc=True)
    if (p>d).any(): raise ValueError('official data look-ahead detected')
    return True

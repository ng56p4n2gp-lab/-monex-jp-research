"""Build leakage-safe predictions for V13 daily simulation.
Fits only on rows strictly before each evaluation block. Future MFE/MAE columns are
labels and are never model inputs. Research only; no broker calls.
"""
from dataclasses import dataclass
import numpy as np
import pandas as pd

FEATURES=['ret1','ret3','volratio','vwap_gap','close_pos','range_pct','body_pct','minute']
LABEL_MFE='mfe60'; LABEL_MAE='mae60'

@dataclass(frozen=True)
class Config:
    min_train_rows:int=5000
    ridge:float=1e-3
    upside_threshold:float=.01
    crash_threshold:float=.02

class RidgeModel:
    def __init__(self,alpha=1e-3): self.alpha=alpha
    def fit(self,X,y):
        self.mu=np.nanmean(X,axis=0); self.sd=np.nanstd(X,axis=0); self.sd[self.sd==0]=1
        Z=np.nan_to_num((X-self.mu)/self.sd); A=Z.T@Z+self.alpha*np.eye(Z.shape[1])
        self.b=np.linalg.solve(A,Z.T@np.nan_to_num(y)); self.intercept=float(np.nanmean(y)); return self
    def predict(self,X): return np.nan_to_num((X-self.mu)/self.sd)@self.b+self.intercept

def sigmoid(x): return 1/(1+np.exp(-np.clip(x,-30,30)))

def build_walk_forward(df:pd.DataFrame,cfg=Config()):
    d=df.copy(); d['decision_at']=pd.to_datetime(d['decision_at']); d=d.sort_values('decision_at')
    forbidden={LABEL_MFE,LABEL_MAE,'mfe5','mfe10','mfe15','mfe30','mae5','mae10','mae15','mae30','future_end'}
    assert not (set(FEATURES)&forbidden), 'future label leaked into FEATURES'
    rows=[]
    for day,g in d.groupby(d['decision_at'].dt.date,sort=True):
        cutoff=g['decision_at'].min(); train=d[d['decision_at']<cutoff]
        if len(train)<cfg.min_train_rows: continue
        X=train[FEATURES].to_numpy(float); Xt=g[FEATURES].to_numpy(float)
        mfe=RidgeModel(cfg.ridge).fit(X,train[LABEL_MFE].to_numpy(float)).predict(Xt)
        mae=RidgeModel(cfg.ridge).fit(X,train[LABEL_MAE].to_numpy(float)).predict(Xt)
        # Conservative probability proxies calibrated from predicted margin vs thresholds.
        scale=max(.002,float(np.nanstd(train[LABEL_MFE])))
        p_up=sigmoid((mfe-cfg.upside_threshold)/scale)
        crash_scale=max(.002,float(np.nanstd(train[LABEL_MAE])))
        p_crash=sigmoid((mae-cfg.crash_threshold)/crash_scale)
        out=g[['decision_at','symbol',LABEL_MFE,LABEL_MAE]].copy()
        out['day']=str(day); out['predicted_mfe']=mfe; out['predicted_mae']=np.maximum(0,mae)
        out['upside_probability']=p_up; out['crash_probability']=p_crash
        out['future_mfe']=g[LABEL_MFE].to_numpy(); out['future_mae']=g[LABEL_MAE].to_numpy()
        out['future_end']=g.get('end60',pd.Series(np.zeros(len(g)),index=g.index)).to_numpy()
        rows.append(out)
    return pd.concat(rows,ignore_index=True) if rows else pd.DataFrame()

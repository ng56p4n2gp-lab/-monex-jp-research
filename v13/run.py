"""V13 predictive MFE/MAE learning. Research only; no live orders."""
import importlib.util,json,time
from pathlib import Path
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1];spec=importlib.util.spec_from_file_location('v7',ROOT/'v7'/'run.py');v7=importlib.util.module_from_spec(spec);spec.loader.exec_module(v7)
OUT=Path('results_v13');OUT.mkdir(exist_ok=True);MAX_PRICE=2000.;SHARES=100;HORIZONS=(1,2,3,6,12)
def features(sym,d):
 d=d.sort_values('timestamp').copy();idx=pd.DatetimeIndex(pd.to_datetime(d['timestamp'],errors='coerce'));valid=~idx.isna();d=d.loc[valid].copy();idx=idx[valid];d.index=idx
 if len(d)<40:return []
 o,h,l,c,v=[d[x].astype(float) for x in ['open','high','low','close','volume']];day=pd.Series(idx.date,index=idx);mins=np.asarray(idx.hour*60+idx.minute)
 tp=(h+l+c)/3;cv=v.groupby(day).cumsum().replace(0,np.nan);vwap=(tp*v).groupby(day).cumsum()/cv;volbase=v.shift(1).rolling(20,min_periods=10).mean().replace(0,np.nan);rng=(h-l).replace(0,np.nan);ret1=c.pct_change();ret3=c/c.shift(3)-1;rows=[]
 for i in range(20,len(d)-1):
  if not (540<=mins[i]<900) or not (0<c.iloc[i]<=MAX_PRICE):continue
  e=i+1
  if day.iloc[e]!=day.iloc[i] or not (0<o.iloc[e]<=MAX_PRICE):continue
  nums=[ret1.iloc[i],ret3.iloc[i],v.iloc[i]/volbase.iloc[i],c.iloc[i]/vwap.iloc[i]-1,(c.iloc[i]-l.iloc[i])/rng.iloc[i],rng.iloc[i]/c.iloc[i],abs(c.iloc[i]-o.iloc[i])/c.iloc[i]]
  if not all(np.isfinite(x) for x in nums):continue
  base={'symbol':sym,'time':str(idx[i]),'date':str(idx[i].date()),'minute':int(mins[i]),'entry':float(o.iloc[e]),'ret1':float(nums[0]),'ret3':float(nums[1]),'volratio':float(nums[2]),'vwap_gap':float(nums[3]),'closepos':float(nums[4]),'range_pct':float(nums[5]),'body_pct':float(nums[6])}
  for n in HORIZONS:
   js=[j for j in range(e,min(e+n,len(d))) if day.iloc[j]==day.iloc[e] and mins[j]<900]
   if js:
    en=base['entry'];base[f'mfe_{n*5}m']=float(h.iloc[js].max()/en-1);base[f'mae_{n*5}m']=float(max(0,1-l.iloc[js].min()/en));base[f'end_{n*5}m']=float(c.iloc[js[-1]]/en-1)
  if 'mfe_60m' in base:rows.append(base)
 return rows
def main():
 t=time.time();uni=v7.build_universe();codes=uni.code.astype(str).tolist();rows=[];reports=[]
 for k in range(0,len(codes),100):
  bars,rep=v7.download(codes[k:k+100],period='60d',interval='5m',batch_size=20,pause=.5,retries=3);reports.append(rep)
  for sym,d in bars.groupby('symbol',sort=False):rows.extend(features(sym,d))
  print('chunk',k//100+1,'samples',len(rows),flush=True)
 df=pd.DataFrame(rows);df.to_csv(OUT/'learning_samples.csv',index=False);pd.concat(reports,ignore_index=True).to_csv(OUT/'download_report.csv',index=False)
 if df.empty:raise RuntimeError('no samples')
 dates=sorted(df.date.unique());a=dates[int(len(dates)*.60)];b=dates[int(len(dates)*.80)];df['split']=np.where(df.date<a,'train',np.where(df.date<b,'selection','holdout'))
 df['vol_bucket']=pd.cut(df.volratio,[-np.inf,1,1.5,2,3,np.inf]);df['ret3_bucket']=pd.cut(df.ret3,[-np.inf,-.02,-.01,-.005,0,.01,np.inf]);df['vwap_bucket']=pd.cut(df.vwap_gap,[-np.inf,-.01,-.003,0,.003,.01,np.inf]);keys=['minute','vol_bucket','ret3_bucket','vwap_bucket']
 g=df.groupby(keys+['split'],observed=True).agg(samples=('symbol','size'),mfe60=('mfe_60m','mean'),mae60=('mae_60m','mean'),p1=('mfe_60m',lambda x:(x>=.01).mean()),p3=('mfe_60m',lambda x:(x>=.03).mean()),p6=('mfe_60m',lambda x:(x>=.06).mean()),p10=('mfe_60m',lambda x:(x>=.10).mean()),crash2=('mae_60m',lambda x:(x>=.02).mean())).reset_index();g.to_csv(OUT/'bucket_learning.csv',index=False)
 sel=g[(g.split=='selection')&(g.samples>=50)].copy();sel['score']=sel.p6*6+sel.p3*3+sel.p1-sel.crash2*8-sel.mae60*4;top=sel.sort_values(['score','samples'],ascending=False).head(100);hold=g[g.split=='holdout'].merge(top[keys],on=keys,how='inner').sort_values(['p6','p3','p1','crash2'],ascending=[False,False,False,True]);hold.to_csv(OUT/'holdout_top.csv',index=False)
 summary={'engine':'V13_FORWARD_MFE_MAE_LEARNING','max_price':MAX_PRICE,'shares':SHARES,'horizons_minutes':[5,10,15,30,60],'samples':len(df),'selection_start':a,'holdout_start':b,'fees_included':False,'tax_included':False,'live_trading_enabled':False,'runtime_seconds':round(time.time()-t,1)};(OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8');(OUT/'RESULT_JP.md').write_text('# V13 先回り学習\n\n未来情報を使わず次足始値からMFE/MAEを検証。\n\n'+hold.head(30).to_markdown(index=False),encoding='utf-8');print(json.dumps(summary,ensure_ascii=False,indent=2))
if __name__=='__main__':main()

"""V11: compare five short-term long-entry patterns on 5m JPX data.
Gross target reach only: fees/taxes intentionally excluded. Research only.
"""
import importlib.util, json, time
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('v7',ROOT/'v7'/'run.py'); v7=importlib.util.module_from_spec(spec); spec.loader.exec_module(v7)
OUT=Path('results_v11'); OUT.mkdir(exist_ok=True); TARGETS=(300,500,1000); SHARES=100

def analyze_symbol(sym,d):
 d=d.sort_values('timestamp').copy(); d=d.set_index(pd.to_datetime(d.timestamp));
 if len(d)<30:return []
 o,h,l,c,v=[d[x].astype(float) for x in ['open','high','low','close','volume']]
 day=pd.Series(d.index.date,index=d.index); typical=(h+l+c)/3; cumv=v.groupby(day).cumsum().replace(0,np.nan); vwap=(typical*v).groupby(day).cumsum()/cumv
 volratio=v/v.shift(1).rolling(20,min_periods=10).mean().replace(0,np.nan); high12=h.shift(1).rolling(12,min_periods=6).max(); ret3=c/c.shift(3)-1
 body=c-o; lower=np.minimum(o,c)-l; upper=h-np.maximum(o,c)
 signals={
 'volume_breakout':(c>high12)&(volratio>=2),
 'vwap_reclaim_volume':(c>vwap)&(c.shift(1)<=vwap.shift(1))&(volratio>=1.5),
 'opening_drop_vwap_reclaim':(ret3<=-.01)&(c>vwap)&(d.index.hour<=10),
 'long_lower_wick_volume':(c>o)&(lower>=abs(body)*1.5)&(lower>upper)&(volratio>=1.5),
 'high_breakout_first_pullback':(c.shift(1)>high12.shift(1))&(l<=high12)&(c>=high12)}
 rows=[]
 for name,sig in signals.items():
  for i in np.flatnonzero(sig.fillna(False).to_numpy()):
   entry=float(c.iloc[i]); mi=d.index[i].hour*60+d.index[i].minute; sess=0 if 540<=mi<=690 else 1 if 750<=mi<=930 else -1
   for win,nbar in ((5,1),(10,2)):
    js=[]
    for j in range(i+1,min(i+nbar+1,len(d))):
     mj=d.index[j].hour*60+d.index[j].minute; sj=0 if 540<=mj<=690 else 1 if 750<=mj<=930 else -1
     if sj==sess and (d.index[j]-d.index[i]).total_seconds()<=win*60:js.append(j)
    if not js:continue
    best=float(h.iloc[js].max()); end=float(c.iloc[js[-1]]); row={'symbol':sym,'time':str(d.index[i]),'date':str(d.index[i].date()),'strategy':name,'window_min':win,'entry':entry,'gross_best':(best-entry)*SHARES,'gross_end':(end-entry)*SHARES}
    for t in TARGETS:row[f'hit_{t}']=row['gross_best']>=t
    rows.append(row)
 return rows

def main():
 t0=time.time(); uni=v7.build_universe(); codes=uni.code.astype(str).tolist(); allrows=[]; reports=[]
 for k in range(0,len(codes),100):
  chunk=codes[k:k+100]; bars,rep=v7.download(chunk,period='60d',interval='5m',batch_size=20,pause=.5,retries=3); reports.append(rep)
  for sym,d in bars.groupby('symbol',sort=False):allrows.extend(analyze_symbol(sym,d))
  print('chunk',k//100+1,'bars',len(bars),'signals',len(allrows),flush=True)
 tr=pd.DataFrame(allrows); tr.to_csv(OUT/'trades.csv',index=False); pd.concat(reports,ignore_index=True).to_csv(OUT/'download_report.csv',index=False)
 if tr.empty:raise RuntimeError('no trades')
 dates=sorted(tr.date.unique()); cut=dates[int(len(dates)*.75)]; tr['split']=np.where(tr.date<cut,'train','validation')
 agg=tr.groupby(['strategy','window_min','split']).agg(samples=('symbol','size'),avg_gross_end=('gross_end','mean'),**{f'hit_{t}':(f'hit_{t}','mean') for t in TARGETS}).reset_index(); agg.to_csv(OUT/'ranking.csv',index=False)
 val=agg[agg.split=='validation'].sort_values(['hit_500','avg_gross_end'],ascending=False)
 summary={'engine':'V11_FIVE_STRATEGIES_GROSS','symbols':len(codes),'trades':len(tr),'validation_start':cut,'target_primary_yen':500,'fees_included':False,'tax_included':False,'live_trading_enabled':False,'runtime_seconds':round(time.time()-t0,1)}
 (OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8'); (OUT/'RESULT_JP.md').write_text('# V11 5パターン比較\n\n手数料・税金なし。+500円到達率を主ランキング。\n\n'+val.to_markdown(index=False),encoding='utf-8'); print(json.dumps(summary,ensure_ascii=False,indent=2)); print(val.to_string(index=False))
if __name__=='__main__':main()

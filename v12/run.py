"""V12: refine 10-minute +500 JPY signals for <=200,000 JPY positions.
Research only. Fees/taxes excluded per current specification.
Signal is formed on a completed bar; entry is NEXT BAR OPEN to reduce look-ahead.
"""
import importlib.util, json, math, time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('v7',ROOT/'v7'/'run.py'); v7=importlib.util.module_from_spec(spec); spec.loader.exec_module(v7)
OUT=Path('results_v12'); OUT.mkdir(exist_ok=True)
MAX_POSITION=200_000; SHARES=100; MAX_PRICE=MAX_POSITION/SHARES; TARGET=500

def wilson_lower(k,n,z=1.96):
    if n<=0:return 0.0
    p=k/n; den=1+z*z/n
    return (p+z*z/(2*n)-z*math.sqrt((p*(1-p)+z*z/(4*n))/n))/den

def analyze(sym,d):
    d=d.sort_values('timestamp').copy(); idx=pd.to_datetime(d.timestamp); d.index=idx
    if len(d)<40:return []
    o,h,l,c,v=[d[x].astype(float) for x in ['open','high','low','close','volume']]
    day=pd.Series(idx.date,index=idx); typical=(h+l+c)/3
    cv=v.groupby(day).cumsum().replace(0,np.nan); vw=(typical*v).groupby(day).cumsum()/cv
    basevol=v.shift(1).rolling(20,min_periods=10).mean().replace(0,np.nan); vr=v/basevol
    r3=c/c.shift(3)-1; body=(c-o).abs(); rng=(h-l).replace(0,np.nan); closepos=(c-l)/rng
    cross=(c.shift(1)<=vw.shift(1))&(c>vw)
    minutes=idx.hour*60+idx.minute; opening=(minutes>=540)&(minutes<=600)
    rows=[]
    # Broad grid; all conditions observable at completed signal bar.
    for drop in (.003,.005,.0075,.01,.015,.02):
      for volmin in (0.8,1.0,1.25,1.5,2.0,3.0):
       for cpmin in (.5,.6,.7,.8):
        sig=opening&cross&(r3<=-drop)&(vr>=volmin)&(closepos>=cpmin)&(c<=MAX_PRICE)
        for i in np.flatnonzero(sig.fillna(False).to_numpy()):
          e=i+1
          if e>=len(d) or day.iloc[e]!=day.iloc[i]:continue
          em=minutes[e]
          if not (540<=em<690):continue
          entry=float(o.iloc[e])
          if entry<=0 or entry>MAX_PRICE:continue
          js=[]
          for j in range(e,min(e+2,len(d))):
            if day.iloc[j]==day.iloc[e] and minutes[j]<690:js.append(j)
          if not js:continue
          best=float(h.iloc[js].max()); end=float(c.iloc[js[-1]])
          rows.append({'symbol':sym,'signal_time':str(idx[i]),'date':str(idx[i].date()),'entry':entry,'drop':drop,'volmin':volmin,'closepos':cpmin,'signal_minute':int(minutes[i]),'volratio':float(vr.iloc[i]),'r3':float(r3.iloc[i]),'vwap_recovery':float(c.iloc[i]/vw.iloc[i]-1),'gross_best':(best-entry)*SHARES,'gross_end':(end-entry)*SHARES,'hit':(best-entry)*SHARES>=TARGET})
    return rows

def main():
 t0=time.time(); uni=v7.build_universe(); codes=uni.code.astype(str).tolist(); rows=[]; reps=[]
 for k in range(0,len(codes),100):
  bars,rep=v7.download(codes[k:k+100],period='60d',interval='5m',batch_size=20,pause=.5,retries=3); reps.append(rep)
  for sym,d in bars.groupby('symbol',sort=False):rows.extend(analyze(sym,d))
  print('chunk',k//100+1,'rows',len(rows),flush=True)
 tr=pd.DataFrame(rows); tr.to_csv(OUT/'trades.csv',index=False); pd.concat(reps,ignore_index=True).to_csv(OUT/'download_report.csv',index=False)
 if tr.empty:raise RuntimeError('no V12 signals')
 dates=sorted(tr.date.unique()); cut=dates[int(len(dates)*.75)]; tr['split']=np.where(tr.date<cut,'train','validation')
 g=tr.groupby(['drop','volmin','closepos','signal_minute','split']).agg(samples=('hit','size'),hits=('hit','sum'),hit_rate=('hit','mean'),avg_gross_end=('gross_end','mean')).reset_index(); g['wilson95_lower']=g.apply(lambda r:wilson_lower(int(r.hits),int(r.samples)),axis=1)
 p=g.pivot(index=['drop','volmin','closepos','signal_minute'],columns='split',values=['samples','hit_rate','avg_gross_end','wilson95_lower']).reset_index(); p.columns=['_'.join([str(x) for x in col if str(x)]) for col in p.columns]
 # Robust candidates: enough independent observations and strong in BOTH periods.
 for col in ['samples_train','samples_validation','hit_rate_train','hit_rate_validation','wilson95_lower_validation']: 
  if col not in p:p[col]=0
 p['robust_rate']=p[['hit_rate_train','hit_rate_validation']].min(axis=1); p=p[(p.samples_train>=300)&(p.samples_validation>=100)].sort_values(['robust_rate','wilson95_lower_validation','samples_validation'],ascending=False)
 p.to_csv(OUT/'ranking.csv',index=False)
 summary={'engine':'V12_SUB200K_SIGNAL_REFINEMENT','max_position_yen':MAX_POSITION,'max_price_for_100_shares':MAX_PRICE,'target_gross_yen':TARGET,'entry':'next_bar_open','fees_included':False,'tax_included':False,'live_trading_enabled':False,'rows':len(tr),'validation_start':cut,'best_robust_rate':None if p.empty else float(p.iloc[0].robust_rate),'runtime_seconds':round(time.time()-t0,1)}
 (OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8'); (OUT/'RESULT_JP.md').write_text('# V12 20万円以内シグナル精査\n\n完成足で判定し、次の5分足始値でエントリー。10分以内+500円到達を検証。\n\n'+p.head(30).to_markdown(index=False),encoding='utf-8'); print(json.dumps(summary,ensure_ascii=False,indent=2)); print(p.head(30).to_string(index=False))
if __name__=='__main__':main()

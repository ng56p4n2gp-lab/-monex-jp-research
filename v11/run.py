"""V11: compare five short-term long-entry patterns on 5m JPX data.
Gross target reach only: fees/taxes intentionally excluded per current specification.
Research only; no live orders.
"""
import importlib.util, json, time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('v7', ROOT/'v7'/'run.py')
v7=importlib.util.module_from_spec(spec); spec.loader.exec_module(v7)
OUT=Path('results_v11'); OUT.mkdir(exist_ok=True)
TARGETS=(300,500,1000); SHARES=100


def same_session(idx,a,b):
    h=idx.hour*60+idx.minute
    am=(h>=540)&(h<=690); pm=(h>=750)&(h<=930)
    return (am & np.roll(am,a) & np.roll(am,b)) | (pm & np.roll(pm,a) & np.roll(pm,b))


def analyze_symbol(sym, d):
    d=d.sort_index().copy()
    if len(d)<30:return []
    o,h,l,c,v=[d[x].astype(float) for x in ['Open','High','Low','Close','Volume']]
    typical=(h+l+c)/3
    day=pd.Series(d.index.date,index=d.index)
    cumv=v.groupby(day).cumsum().replace(0,np.nan)
    vwap=(typical*v).groupby(day).cumsum()/cumv
    vol20=v.shift(1).rolling(20,min_periods=10).mean()
    volratio=v/vol20
    high12=h.shift(1).rolling(12,min_periods=6).max()
    prevday_high=h.groupby(day).transform('max').groupby(day).shift(1)  # only used as loose context
    ret3=c/c.shift(3)-1
    body=c-o; rng=(h-l).replace(0,np.nan)
    lower=np.minimum(o,c)-l
    upper=h-np.maximum(o,c)

    signals={
      'volume_breakout': (c>high12)&(volratio>=2.0),
      'vwap_reclaim_volume': (c>vwap)&(c.shift(1)<=vwap.shift(1))&(volratio>=1.5),
      'opening_drop_vwap_reclaim': (ret3<=-0.01)&(c>vwap)&(d.index.hour<=10),
      'long_lower_wick_volume': (c>o)&(lower>=abs(body)*1.5)&(lower>upper)&(volratio>=1.5),
      'high_breakout_first_pullback': (c.shift(1)>high12.shift(1))&(l<=high12)&(c>=high12),
    }
    rows=[]
    for name,sig in signals.items():
      pos=np.flatnonzero(sig.fillna(False).to_numpy())
      for i in pos:
        entry=float(c.iloc[i])
        for win,nbar in ((5,1),(10,2)):
          js=[j for j in range(i+1,min(i+nbar+1,len(d))) if (d.index[j]-d.index[i]).total_seconds()<=win*60]
          if not js: continue
          # prohibit crossing lunch/session boundary
          mi=d.index[i].hour*60+d.index[i].minute
          sess=0 if 540<=mi<=690 else 1 if 750<=mi<=930 else -1
          js=[j for j in js if (0 if 540<=d.index[j].hour*60+d.index[j].minute<=690 else 1 if 750<=d.index[j].hour*60+d.index[j].minute<=930 else -1)==sess]
          if not js: continue
          best=float(h.iloc[js].max())
          end=float(c.iloc[js[-1]])
          gross_best=(best-entry)*SHARES
          gross_end=(end-entry)*SHARES
          row={'symbol':sym,'time':str(d.index[i]),'date':str(d.index[i].date()),'strategy':name,'window_min':win,'entry':entry,'gross_best':gross_best,'gross_end':gross_end}
          for t in TARGETS: row[f'hit_{t}']=gross_best>=t
          rows.append(row)
    return rows


def main():
    t0=time.time(); universe=v7.load_universe(); allrows=[]; report=[]
    symbols=[f'{x}.T' for x in universe]
    for k in range(0,len(symbols),20):
      batch=symbols[k:k+20]
      data=v7.download_batch(batch, period='60d', interval='5m')
      for s in batch:
        try:
          d=v7.extract_symbol(data,s)
          report.append({'symbol':s,'rows':len(d),'status':'ok' if len(d) else 'missing'})
          if len(d): allrows.extend(analyze_symbol(s,d))
        except Exception as e: report.append({'symbol':s,'rows':0,'status':type(e).__name__})
      time.sleep(.5)
    tr=pd.DataFrame(allrows); tr.to_csv(OUT/'trades.csv',index=False)
    pd.DataFrame(report).to_csv(OUT/'download_report.csv',index=False)
    if tr.empty: raise RuntimeError('no trades')
    dates=sorted(tr.date.unique()); cut=dates[int(len(dates)*.75)]
    tr['split']=np.where(tr.date<cut,'train','validation')
    agg=tr.groupby(['strategy','window_min','split']).agg(samples=('symbol','size'),avg_gross_end=('gross_end','mean'),**{f'hit_{t}':(f'hit_{t}','mean') for t in TARGETS}).reset_index()
    agg.to_csv(OUT/'ranking.csv',index=False)
    val=agg[agg.split=='validation'].sort_values(['hit_500','avg_gross_end'],ascending=False)
    summary={'engine':'V11_FIVE_STRATEGIES_GROSS','symbols':len(universe),'trades':len(tr),'validation_start':cut,'target_primary_yen':500,'fees_included':False,'tax_included':False,'live_trading_enabled':False,'runtime_seconds':round(time.time()-t0,1)}
    (OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    (OUT/'RESULT_JP.md').write_text('# V11 5パターン比較\n\n手数料・税金を含めないグロス利益で判定。+500円到達率を主ランキングに使用。\n\n'+val.to_markdown(index=False),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2)); print(val.to_string(index=False))
if __name__=='__main__': main()

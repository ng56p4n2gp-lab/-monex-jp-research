from __future__ import annotations
import argparse, json, re, time
from io import BytesIO
from pathlib import Path
from urllib.parse import urljoin

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

JPX_LIST_PAGE='https://www.jpx.co.jp/markets/statistics-equities/misc/01.html'
COLS=['timestamp','symbol','open','high','low','close','volume']
GROUP=['window_min','breakout12','volume_bucket','vwap_bucket','range_bucket','time_bucket','weekday']
FEE_BOUNDS=np.array([50_000,100_000,200_000,500_000,1_000_000,1_500_000,30_000_000],float)
FEE_VALUES=np.array([55,99,115,275,535,640,1013,1070],float)

def fee_vec(v):
    return FEE_VALUES[np.searchsorted(FEE_BOUNDS,np.asarray(v,float),side='left')]

def net_vec(buy_price,sell_price,shares,tax_rate=.20315):
    buy=np.asarray(buy_price,float)*shares; sell=np.asarray(sell_price,float)*shares
    pre=sell-buy-fee_vec(buy)-fee_vec(sell)
    return pre-np.maximum(pre,0)*tax_rate

def find_col(cols,words):
    for c in cols:
        s=str(c).strip().lower()
        if all(w.lower() in s for w in words): return c
    for c in cols:
        s=str(c).strip().lower()
        if any(w.lower() in s for w in words): return c
    return None

def build_universe():
    s=requests.Session(); h={'User-Agent':'Mozilla/5.0'}
    r=s.get(JPX_LIST_PAGE,timeout=30,headers=h); r.raise_for_status()
    soup=BeautifulSoup(r.text,'html.parser')
    links=[urljoin(JPX_LIST_PAGE,a['href']) for a in soup.find_all('a',href=True) if re.search(r'\.(xlsx?|xls)(?:$|\?)',a['href'],re.I)]
    pref=[u for u in links if any(k in u.lower() for k in ['data_j','list','stock','misc'])]
    if not (pref or links): raise RuntimeError('JPX listed-issues Excel not found')
    u=(pref or links)[0]
    rr=s.get(u,timeout=60,headers=h); rr.raise_for_status(); df=pd.read_excel(BytesIO(rr.content))
    code=find_col(df.columns,['コード']) or find_col(df.columns,['code'])
    market=find_col(df.columns,['市場']) or find_col(df.columns,['market'])
    product=find_col(df.columns,['商品']) or find_col(df.columns,['product'])
    x=pd.DataFrame({'code':df[code].astype(str).str.extract(r'([0-9A-Z]{4,5})',expand=False)})
    x['market']=df[market].astype(str) if market is not None else ''
    x['product']=df[product].astype(str) if product is not None else ''
    if market is not None:
        x=x[x.market.str.contains('プライム|スタンダード|グロース|Prime|Standard|Growth',case=False,regex=True,na=False)]
    if product is not None:
        x=x[~x['product'].str.contains(r'ETF|ETN|REIT|投資法人|受益証券|出資証券|優先|インフラ',case=False,regex=True,na=False)]
    return x[x.code.str.fullmatch(r'\d{4}',na=False)].drop_duplicates('code').sort_values('code').reset_index(drop=True)

def normalize(df,sym):
    if df is None or len(df)==0: return pd.DataFrame(columns=COLS)
    x=df.reset_index(); dt=x.columns[0]
    x=x.rename(columns={dt:'timestamp','Open':'open','High':'high','Low':'low','Close':'close','Volume':'volume'})
    x['symbol']=sym.replace('.T','')
    if any(c not in x.columns for c in COLS): return pd.DataFrame(columns=COLS)
    x=x[COLS]; x['timestamp']=pd.to_datetime(x.timestamp,errors='coerce')
    for c in ['open','high','low','close','volume']: x[c]=pd.to_numeric(x[c],errors='coerce')
    return x.dropna(subset=['timestamp','open','high','low','close'])

def download(symbols,period='60d',interval='5m',batch_size=20,pause=.5,retries=3):
    import yfinance as yf
    frames=[]; report=[]; syms=[f'{s}.T' for s in symbols]
    for i in range(0,len(syms),batch_size):
        batch=syms[i:i+batch_size]; raw=None; err=''
        for a in range(1,retries+1):
            try:
                raw=yf.download(tickers=' '.join(batch),period=period,interval=interval,group_by='ticker',auto_adjust=False,actions=False,threads=False,progress=False,prepost=False,timeout=30)
                if raw is not None and len(raw): break
            except Exception as e: err=str(e)
            time.sleep(pause*a)
        for sym in batch:
            f=pd.DataFrame(columns=COLS)
            try:
                if raw is not None and len(raw):
                    if len(batch)==1: f=normalize(raw,sym)
                    elif hasattr(raw.columns,'levels') and sym in raw.columns.get_level_values(0): f=normalize(raw[sym],sym)
            except Exception as e: err=str(e)
            if len(f):
                frames.append(f); report.append({'symbol':sym[:-2],'status':'ok','rows':len(f),'first_timestamp':f.timestamp.min(),'last_timestamp':f.timestamp.max(),'error':''})
            else: report.append({'symbol':sym[:-2],'status':'missing','rows':0,'first_timestamp':'','last_timestamp':'','error':err})
        time.sleep(pause)
    bars=pd.concat(frames,ignore_index=True).sort_values(['symbol','timestamp']).drop_duplicates(['symbol','timestamp']) if frames else pd.DataFrame(columns=COLS)
    return bars,pd.DataFrame(report)

def add_features(df):
    x=df.sort_values(['symbol','timestamp']).copy(); g=x.groupby('symbol',group_keys=False)
    x['ret_1']=g.close.pct_change(); x['ret_2']=g.close.pct_change(2); x['ret_3']=g.close.pct_change(3)
    x['vol_ma12']=g.volume.transform(lambda s:s.rolling(12,min_periods=5).mean()); x['volume_ratio']=x.volume/x.vol_ma12.replace(0,np.nan)
    d=pd.to_datetime(x.timestamp); day=d.dt.date; typ=(x.high+x.low+x.close)/3; pv=typ*x.volume
    x['vwap']=pv.groupby([x.symbol,day]).cumsum()/x.volume.groupby([x.symbol,day]).cumsum().replace(0,np.nan)
    x['vwap_gap']=x.close/x.vwap-1; x['prev_high12']=g.high.transform(lambda s:s.shift(1).rolling(12,min_periods=5).max())
    x['breakout12']=(x.close>x.prev_high12).astype(int); x['range_pct']=(x.high-x.low)/x.close.replace(0,np.nan)
    x['minute_of_day']=d.dt.hour*60+d.dt.minute; x['weekday']=d.dt.weekday
    m=x.minute_of_day; date=d.dt.strftime('%Y-%m-%d'); x['session_id']=pd.NA
    x.loc[(m>=540)&(m<=690),'session_id']=date[(m>=540)&(m<=690)]+'-AM'; x.loc[(m>=750)&(m<=930),'session_id']=date[(m>=750)&(m<=930)]+'-PM'
    return x[x.session_id.notna()].copy()

def backtest_chunk(df,shares=100,target=1000,slippage=1.0):
    x=add_features(df); parts=[]
    for sym,g in x.groupby('symbol',sort=False):
        g=g.sort_values('timestamp').reset_index(drop=True); n=len(g)
        if n<2: continue
        ts=pd.to_datetime(g.timestamp); sid=g.session_id; entry=g.close.to_numpy(float)+slippage
        hi=g.high.to_numpy(float)-slippage; lo=g.low.to_numpy(float)-slippage; cl=g.close.to_numpy(float)-slippage
        for w in (5,10):
            ts1=ts.shift(-1); valid1=(sid.shift(-1)==sid)&(ts1>ts)&((ts1-ts)<=pd.Timedelta(minutes=w))
            if w==10:
                ts2=ts.shift(-2); valid2=(sid.shift(-2)==sid)&(ts2>ts)&((ts2-ts)<=pd.Timedelta(minutes=w))
            else:
                ts2=pd.Series(pd.NaT,index=ts.index); valid2=pd.Series(False,index=ts.index)
            hi1=np.where(valid1,np.roll(hi,-1),np.nan); lo1=np.where(valid1,np.roll(lo,-1),np.nan); cl1=np.where(valid1,np.roll(cl,-1),np.nan)
            hi2=np.where(valid2,np.roll(hi,-2),np.nan); lo2=np.where(valid2,np.roll(lo,-2),np.nan); cl2=np.where(valid2,np.roll(cl,-2),np.nan)
            sell_hi=np.fmax(hi1,hi2) if w==10 else hi1; sell_lo=np.fmin(lo1,lo2) if w==10 else lo1; end_sell=np.where(valid2,cl2,cl1) if w==10 else cl1
            best=net_vec(entry,np.maximum(sell_hi,0),shares); worst=net_vec(entry,np.maximum(sell_lo,0),shares); end=net_vec(entry,np.maximum(end_sell,0),shares)
            h1=net_vec(entry,np.maximum(hi1,0),shares); h2=net_vec(entry,np.maximum(hi2,0),shares) if w==10 else np.full(n,np.nan)
            hit=(valid1.to_numpy()&(h1>=target)) | (valid2.to_numpy()&(h2>=target))
            anyv=valid1.to_numpy()|valid2.to_numpy(); idx=np.flatnonzero(anyv)
            if not len(idx): continue
            r=pd.DataFrame({'entry_time':ts.iloc[idx].to_numpy(),'window_min':w,'target_hit':hit[idx],'best_net_profit':best[idx],'worst_net_profit':worst[idx],'end_net_profit':end[idx]})
            for c in ['volume_ratio','vwap_gap','breakout12','range_pct','minute_of_day','weekday']: r[c]=g[c].iloc[idx].to_numpy()
            parts.append(r)
    return pd.concat(parts,ignore_index=True) if parts else pd.DataFrame()

def bucket_and_daily(r):
    if r.empty: return pd.DataFrame()
    z=r.copy(); z['entry_date']=pd.to_datetime(z.entry_time).dt.date.astype(str)
    z['volume_bucket']=pd.cut(z.volume_ratio,[-1,1,1.5,2,3,5,10,float('inf')]).astype(str)
    z['vwap_bucket']=pd.cut(z.vwap_gap,[-float('inf'),-.02,-.01,-.005,0,.005,.01,.02,float('inf')]).astype(str)
    z['range_bucket']=pd.cut(z.range_pct,[-1,.0025,.005,.01,.02,.04,float('inf')]).astype(str)
    z['time_bucket']=pd.cut(z.minute_of_day,[0,555,570,600,660,690,750,810,870,930,1440],include_lowest=True).astype(str)
    q=(z.groupby(['entry_date']+GROUP,dropna=False,observed=True)
       .agg(samples=('target_hit','size'),hits=('target_hit','sum'),sum_best=('best_net_profit','sum'),sum_worst=('worst_net_profit','sum'),sum_end=('end_net_profit','sum')).reset_index())
    return q

def strict100_from_daily(daily,min_train=50,min_valid=15,max_avg_adverse=1200,folds=3):
    if daily.empty: return pd.DataFrame(),[]
    daily=(daily.groupby(['entry_date']+GROUP,dropna=False,as_index=False)[['samples','hits','sum_best','sum_worst','sum_end']].sum())
    dates=sorted(daily.entry_date.unique())
    if len(dates)<20: return pd.DataFrame(),[]
    base=max(10,int(len(dates)*.55)); fold_size=max(3,(len(dates)-base)//folds); merged=None; used=[]
    def summ(d,p):
        q=d.groupby(GROUP,dropna=False,as_index=False)[['samples','hits','sum_best','sum_worst','sum_end']].sum()
        q[p+'_samples']=q.samples; q[p+'_hit_rate']=q.hits/q.samples; q[p+'_avg_best']=q.sum_best/q.samples; q[p+'_avg_worst']=q.sum_worst/q.samples; q[p+'_avg_end']=q.sum_end/q.samples
        return q[GROUP+[p+'_samples',p+'_hit_rate',p+'_avg_best',p+'_avg_worst',p+'_avg_end']]
    for k in range(folds):
        te=base+k*fold_size; ve=len(dates) if k==folds-1 else min(len(dates),te+fold_size)
        if ve<=te: continue
        tr=daily[daily.entry_date.isin(dates[:te])]; va=daily[daily.entry_date.isin(dates[te:ve])]
        a=summ(tr,f'f{k}_train'); b=summ(va,f'f{k}_valid'); m=a.merge(b,on=GROUP,how='inner')
        m=m[(m[f'f{k}_train_samples']>=min_train)&(m[f'f{k}_valid_samples']>=min_valid)&(m[f'f{k}_train_hit_rate']==1.0)&(m[f'f{k}_valid_hit_rate']==1.0)&(m[f'f{k}_valid_avg_worst']>=-abs(max_avg_adverse))]
        merged=m if merged is None else merged.merge(m,on=GROUP,how='inner'); used.append((dates[0],dates[te-1],dates[te],dates[ve-1]))
    if merged is None or merged.empty: return pd.DataFrame(),used
    cols=[c for c in merged if c.endswith('_valid_samples')]; merged['validation_samples_total']=merged[cols].sum(axis=1)
    return merged.sort_values('validation_samples_total',ascending=False),used

def self_test():
    assert fee_vec([50_000,100_000,200_000,500_000]).tolist()==[55,99,115,275]
    rows=[]
    for sym,base in [('1111',1000.),('2222',2500.)]:
        for i,t in enumerate(pd.date_range('2026-09-10 09:00','2026-09-10 11:30',freq='5min')):
            p=base+i; rows.append([t,sym,p-1,p+3,p-3,p,1000+i*10])
    df=pd.DataFrame(rows,columns=COLS); r=backtest_chunk(df); assert len(r)>0 and set(r.window_min)=={5,10}
    print('SELF_TEST_OK')

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--self-test',action='store_true'); ap.add_argument('--period',default='60d'); ap.add_argument('--interval',default='5m'); ap.add_argument('--batch-size',type=int,default=20); ap.add_argument('--symbol-chunk',type=int,default=100); ap.add_argument('--shares',type=int,default=100); ap.add_argument('--target',type=float,default=1000); ap.add_argument('--slippage',type=float,default=1.0); a=ap.parse_args()
    if a.self_test: self_test(); return
    Path('outputs').mkdir(exist_ok=True); uni=build_universe(); uni.to_csv('outputs/tse_symbols.csv',index=False); codes=uni.code.astype(str).tolist(); print('universe',len(codes),flush=True)
    reports=[]; dailies=[]; total_bars=0; total_bt=0; t_start=time.time()
    for n,start in enumerate(range(0,len(codes),a.symbol_chunk),1):
        chunk=codes[start:start+a.symbol_chunk]; t=time.time(); bars,rep=download(chunk,a.period,a.interval,a.batch_size); rep['chunk']=n; reports.append(rep); total_bars+=len(bars)
        r=backtest_chunk(bars,a.shares,a.target,a.slippage); total_bt+=len(r); d=bucket_and_daily(r); dailies.append(d)
        print(f'chunk {n}: symbols={len(chunk)} bars={len(bars):,} tests={len(r):,} sec={time.time()-t:.1f}',flush=True)
    report=pd.concat(reports,ignore_index=True); report.to_csv('outputs/download_report.csv',index=False)
    daily=pd.concat(dailies,ignore_index=True) if dailies else pd.DataFrame(); candidates,folds=strict100_from_daily(daily); candidates.to_csv('outputs/strict100_candidates.csv',index=False)
    ok=int((report.status=='ok').sum()); missing=int((report.status!='ok').sum()); runtime=time.time()-t_start
    summary={'engine':'V7_FAST_STREAMING','requested_period':a.period,'interval':a.interval,'universe_symbols':len(codes),'download_ok_symbols':ok,'download_missing_symbols':missing,'downloaded_bar_rows':int(total_bars),'backtest_rows':int(total_bt),'strict100_candidate_count':int(len(candidates)),'target_net_profit_yen':a.target,'shares':a.shares,'slippage_yen_each_side_assumption':a.slippage,'runtime_seconds':round(runtime,1),'walk_forward_folds':[{'train_start':x[0],'train_end':x[1],'validation_start':x[2],'validation_end':x[3]} for x in folds],'future_100_percent_guaranteed':False,'live_trading_enabled':False,'decision':'NO_TRADE' if candidates.empty else 'HISTORICAL_CANDIDATES_FOUND'}
    Path('outputs/summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    txt='\n'.join(['# V7 60日高速学習結果',f'- 東証対象銘柄数: {len(codes)}',f'- データ取得成功: {ok}',f'- 取得失敗/欠損: {missing}',f'- 5分足行数: {total_bars:,}',f'- バックテスト判定数: {total_bt:,}',f'- 厳格100%候補数: {len(candidates)}',f'- 実行時間: {runtime/60:.1f}分',f"- 判定: {'買わない' if candidates.empty else '過去100%候補あり（将来保証なし）'}",'', '※ 100%は過去の学習・独立検証区間での結果で、将来保証ではありません。','※ 実売買はOFFです。'])
    Path('outputs/RESULT_JP.md').write_text(txt,encoding='utf-8'); print(txt,flush=True)
if __name__=='__main__': main()

from __future__ import annotations
import argparse, importlib.util, json, time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('v7run', ROOT/'v7'/'run.py')
v7=importlib.util.module_from_spec(spec); assert spec.loader is not None; spec.loader.exec_module(v7)

def pattern_backtest(df, shares=100, slippage=1.0):
    out=[]
    for sym,g in df.groupby('symbol',sort=False):
        g=g.sort_values('timestamp').reset_index(drop=True).copy()
        if len(g)<4: continue
        ts=pd.to_datetime(g.timestamp)
        m=ts.dt.hour*60+ts.dt.minute
        date=ts.dt.strftime('%Y-%m-%d')
        sid=pd.Series(pd.NA,index=g.index,dtype='object')
        am=(m>=540)&(m<=690); pm=(m>=750)&(m<=930)
        sid.loc[am]=date.loc[am]+'-AM'; sid.loc[pm]=date.loc[pm]+'-PM'
        po=g.open.shift(1); pc=g.close.shift(1); ph=g.high.shift(1); pl=g.low.shift(1)
        prev_bear=(pc<po)
        midpoint=(ph+pl)/2.0
        same_prev=(sid.shift(1)==sid)
        touch=(g.low<=midpoint)&(g.high>=midpoint)
        trigger=prev_bear & same_prev & touch & sid.notna()
        for i in np.flatnonzero(trigger.to_numpy()):
            entry=float(midpoint.iloc[i])+slippage
            rec={'symbol':sym,'entry_time':ts.iloc[i],'prev_open':po.iloc[i],'prev_high':ph.iloc[i],'prev_low':pl.iloc[i],'prev_close':pc.iloc[i],'midpoint':midpoint.iloc[i],'entry_price':entry}
            for w,steps in [(5,1),(10,2)]:
                future=[]
                for j in range(i+1,min(len(g),i+steps+1)):
                    if sid.iloc[j]!=sid.iloc[i]: break
                    if ts.iloc[j]-ts.iloc[i] > pd.Timedelta(minutes=w): break
                    future.append(j)
                if not future:
                    rec[f'valid_{w}m']=False; continue
                rec[f'valid_{w}m']=True
                end_sell=float(g.close.iloc[future[-1]])-slippage
                best_sell=float(g.high.iloc[future].max())-slippage
                worst_sell=float(g.low.iloc[future].min())-slippage
                end_net=float(v7.net_vec([entry],[max(end_sell,0)],shares)[0])
                best_net=float(v7.net_vec([entry],[max(best_sell,0)],shares)[0])
                worst_net=float(v7.net_vec([entry],[max(worst_sell,0)],shares)[0])
                rec[f'end_net_{w}m']=end_net
                rec[f'best_net_{w}m']=best_net
                rec[f'worst_net_{w}m']=worst_net
                rec[f'win_{w}m']=end_net>0
                for t in (300,500,1000): rec[f'hit_{t}_{w}m']=best_net>=t
            out.append(rec)
    return pd.DataFrame(out)

def summarize(r):
    rows=[]
    for w in (5,10):
        v=r[r.get(f'valid_{w}m',False)==True].copy() if not r.empty else pd.DataFrame()
        if v.empty:
            rows.append({'window_min':w,'trades':0}); continue
        rows.append({'window_min':w,'trades':len(v),'net_positive_wins':int(v[f'win_{w}m'].sum()),'net_positive_win_rate':float(v[f'win_{w}m'].mean()),'avg_end_net_yen':float(v[f'end_net_{w}m'].mean()),'median_end_net_yen':float(v[f'end_net_{w}m'].median()),'avg_best_net_yen':float(v[f'best_net_{w}m'].mean()),'avg_worst_net_yen':float(v[f'worst_net_{w}m'].mean()),'hit_300_rate':float(v[f'hit_300_{w}m'].mean()),'hit_500_rate':float(v[f'hit_500_{w}m'].mean()),'hit_1000_rate':float(v[f'hit_1000_{w}m'].mean())})
    return pd.DataFrame(rows)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--self-test',action='store_true'); ap.add_argument('--period',default='60d'); ap.add_argument('--interval',default='5m'); ap.add_argument('--batch-size',type=int,default=20); ap.add_argument('--symbol-chunk',type=int,default=100); ap.add_argument('--shares',type=int,default=100); ap.add_argument('--slippage',type=float,default=1.0); a=ap.parse_args()
    if a.self_test:
        rows=[]; times=pd.date_range('2026-09-10 09:00',periods=5,freq='5min'); vals=[(100,101,98,99),(99,100,98.5,99.5),(99.5,102,99,101),(101,103,100,102),(102,104,101,103)]
        for t,(o,h,l,c) in zip(times,vals): rows.append([t,'1111',o,h,l,c,1000])
        r=pattern_backtest(pd.DataFrame(rows,columns=v7.COLS)); assert isinstance(r,pd.DataFrame); print('V10_SELF_TEST_OK'); return
    Path('outputs').mkdir(exist_ok=True); uni=v7.build_universe(); codes=uni.code.astype(str).tolist(); reports=[]; allr=[]; bars_total=0; t0=time.time()
    for n,start in enumerate(range(0,len(codes),a.symbol_chunk),1):
        bars,rep=v7.download(codes[start:start+a.symbol_chunk],a.period,a.interval,a.batch_size); rep['chunk']=n; reports.append(rep); bars_total+=len(bars)
        r=pattern_backtest(bars,a.shares,a.slippage); allr.append(r); print(f'chunk {n}: bars={len(bars):,} pattern_trades={len(r):,}',flush=True)
    report=pd.concat(reports,ignore_index=True); report.to_csv('outputs/download_report.csv',index=False)
    r=pd.concat(allr,ignore_index=True) if allr else pd.DataFrame(); r.to_csv('outputs/midpoint_trades.csv',index=False)
    s=summarize(r); s.to_csv('outputs/summary_by_window.csv',index=False)
    runtime=time.time()-t0; ok=int((report.status=='ok').sum()); miss=int((report.status!='ok').sum())
    summary={'engine':'V10_BEARISH_MIDPOINT_RETRACE','rule':'previous candle bearish only; buy when current candle touches previous (high+low)/2','period':a.period,'interval':a.interval,'universe_symbols':len(codes),'download_ok_symbols':ok,'download_missing_symbols':miss,'downloaded_bar_rows':bars_total,'pattern_trades':int(len(r)),'shares':a.shares,'slippage_yen_each_side':a.slippage,'fees_and_tax_included':True,'runtime_seconds':round(runtime,1),'live_trading_enabled':False,'windows':s.to_dict(orient='records')}
    Path('outputs/summary_v10.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    lines=['# V10 前陰線・値幅中央値戻し買い検証',f'- 対象銘柄数: {len(codes)}',f'- データ取得成功: {ok}',f'- パターン発生数: {len(r):,}',f'- 実行時間: {runtime/60:.1f}分','']
    for _,x in s.iterrows():
        if x.get('trades',0): lines += [f"## {int(x.window_min)}分",f"- 取引数: {int(x.trades):,}",f"- 手数料等控除後の勝率: {x.net_positive_win_rate*100:.2f}%",f"- 平均最終損益: {x.avg_end_net_yen:.0f}円",f"- +300円到達率: {x.hit_300_rate*100:.2f}%",f"- +500円到達率: {x.hit_500_rate*100:.2f}%",f"- +1,000円到達率: {x.hit_1000_rate*100:.2f}%",'']
    lines += ['※ 前足が陽線または同値は除外。','※ 現足が前足の高値・安値中央値に触れた時点で約定したと仮定。','※ 売買手数料・税金・片道1円スリッページを考慮。','※ 将来の勝率を保証しません。実売買はOFFです。']
    Path('outputs/RESULT_V10_JP.md').write_text('\n'.join(lines),encoding='utf-8'); print('\n'.join(lines),flush=True)
if __name__=='__main__': main()

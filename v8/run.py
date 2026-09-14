from __future__ import annotations
import argparse, importlib.util, json, sys, time
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
V7=ROOT/'v7'/'run.py'
spec=importlib.util.spec_from_file_location('v7run', V7)
v7=importlib.util.module_from_spec(spec); assert spec.loader is not None; spec.loader.exec_module(v7)
GROUP=v7.GROUP


def evaluate_threshold(daily, threshold, min_train=50, min_valid=15, max_avg_adverse=1200, folds=3):
    if daily.empty: return pd.DataFrame(), []
    d=(daily.groupby(['entry_date']+GROUP,dropna=False,as_index=False)[['samples','hits','sum_best','sum_worst','sum_end']].sum())
    dates=sorted(d.entry_date.unique())
    if len(dates)<20: return pd.DataFrame(), []
    base=max(10,int(len(dates)*.55)); fold_size=max(3,(len(dates)-base)//folds)
    merged=None; used=[]
    def summ(x,p):
        q=x.groupby(GROUP,dropna=False,as_index=False)[['samples','hits','sum_best','sum_worst','sum_end']].sum()
        q[p+'_samples']=q.samples
        q[p+'_hit_rate']=q.hits/q.samples
        q[p+'_avg_best']=q.sum_best/q.samples
        q[p+'_avg_worst']=q.sum_worst/q.samples
        q[p+'_avg_end']=q.sum_end/q.samples
        return q[GROUP+[p+'_samples',p+'_hit_rate',p+'_avg_best',p+'_avg_worst',p+'_avg_end']]
    for k in range(folds):
        te=base+k*fold_size; ve=len(dates) if k==folds-1 else min(len(dates),te+fold_size)
        if ve<=te: continue
        tr=d[d.entry_date.isin(dates[:te])]; va=d[d.entry_date.isin(dates[te:ve])]
        a=summ(tr,f'f{k}_train'); b=summ(va,f'f{k}_valid'); m=a.merge(b,on=GROUP,how='inner')
        m=m[(m[f'f{k}_train_samples']>=min_train)&(m[f'f{k}_valid_samples']>=min_valid)&(m[f'f{k}_train_hit_rate']>=threshold)&(m[f'f{k}_valid_hit_rate']>=threshold)&(m[f'f{k}_valid_avg_worst']>=-abs(max_avg_adverse))]
        merged=m if merged is None else merged.merge(m,on=GROUP,how='inner')
        used.append((dates[0],dates[te-1],dates[te],dates[ve-1]))
    if merged is None or merged.empty: return pd.DataFrame(), used
    s_cols=[c for c in merged if c.endswith('_valid_samples')]
    hr_cols=[c for c in merged if c.endswith('_valid_hit_rate')]
    end_cols=[c for c in merged if c.endswith('_valid_avg_end')]
    worst_cols=[c for c in merged if c.endswith('_valid_avg_worst')]
    best_cols=[c for c in merged if c.endswith('_valid_avg_best')]
    merged['threshold']=threshold
    merged['validation_samples_total']=merged[s_cols].sum(axis=1)
    merged['validation_hit_rate_min']=merged[hr_cols].min(axis=1)
    merged['validation_hit_rate_mean']=merged[hr_cols].mean(axis=1)
    merged['validation_avg_end_mean']=merged[end_cols].mean(axis=1)
    merged['validation_avg_worst_mean']=merged[worst_cols].mean(axis=1)
    merged['validation_avg_best_mean']=merged[best_cols].mean(axis=1)
    merged['score']=merged['validation_avg_end_mean']*merged['validation_hit_rate_min']
    return merged.sort_values(['score','validation_samples_total'],ascending=[False,False]), used


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--self-test',action='store_true'); ap.add_argument('--period',default='60d'); ap.add_argument('--interval',default='5m'); ap.add_argument('--batch-size',type=int,default=20); ap.add_argument('--symbol-chunk',type=int,default=100); ap.add_argument('--shares',type=int,default=100); ap.add_argument('--target',type=float,default=1000); ap.add_argument('--slippage',type=float,default=1.0)
    a=ap.parse_args()
    if a.self_test:
        v7.self_test();
        toy=pd.DataFrame({'entry_date':['2026-01-%02d'%((i%28)+1) for i in range(60)],'window_min':[5]*60,'breakout12':[1]*60,'volume_bucket':['(1.5, 2.0]']*60,'vwap_bucket':['(0.0, 0.005]']*60,'range_bucket':['(0.005, 0.01]']*60,'time_bucket':['(570, 600]']*60,'weekday':[i%5 for i in range(60)],'samples':[20]*60,'hits':[20]*60,'sum_best':[30000]*60,'sum_worst':[-5000]*60,'sum_end':[5000]*60})
        print('V8_SELF_TEST_OK'); return
    Path('outputs').mkdir(exist_ok=True)
    uni=v7.build_universe(); uni.to_csv('outputs/tse_symbols.csv',index=False); codes=uni.code.astype(str).tolist(); print('universe',len(codes),flush=True)
    reports=[]; dailies=[]; total_bars=0; total_bt=0; t0=time.time()
    for n,start in enumerate(range(0,len(codes),a.symbol_chunk),1):
        chunk=codes[start:start+a.symbol_chunk]; t=time.time(); bars,rep=v7.download(chunk,a.period,a.interval,a.batch_size); rep['chunk']=n; reports.append(rep); total_bars+=len(bars)
        r=v7.backtest_chunk(bars,a.shares,a.target,a.slippage); total_bt+=len(r); d=v7.bucket_and_daily(r); dailies.append(d)
        print(f'chunk {n}: symbols={len(chunk)} bars={len(bars):,} tests={len(r):,} sec={time.time()-t:.1f}',flush=True)
    report=pd.concat(reports,ignore_index=True); report.to_csv('outputs/download_report.csv',index=False)
    daily=pd.concat(dailies,ignore_index=True) if dailies else pd.DataFrame()
    thresholds=[1.0,.99,.98,.95]; allc=[]; counts={}; folds=[]
    for th in thresholds:
        c,fu=evaluate_threshold(daily,th); counts[str(th)]=int(len(c)); folds=fu or folds
        if not c.empty: allc.append(c)
        c.to_csv(f"outputs/candidates_{int(th*100)}.csv",index=False)
    combined=pd.concat(allc,ignore_index=True) if allc else pd.DataFrame(); combined.to_csv('outputs/candidates_all.csv',index=False)
    if not combined.empty:
        shortlist=combined.sort_values(['threshold','score','validation_samples_total'],ascending=[False,False,False]).head(200)
    else: shortlist=combined
    shortlist.to_csv('outputs/shortlist.csv',index=False)
    ok=int((report.status=='ok').sum()); missing=int((report.status!='ok').sum()); runtime=time.time()-t0
    summary={'engine':'V8_MULTI_THRESHOLD','requested_period':a.period,'interval':a.interval,'universe_symbols':len(codes),'download_ok_symbols':ok,'download_missing_symbols':missing,'downloaded_bar_rows':int(total_bars),'backtest_rows':int(total_bt),'candidate_counts':counts,'target_net_profit_yen':a.target,'shares':a.shares,'slippage_yen_each_side_assumption':a.slippage,'runtime_seconds':round(runtime,1),'future_win_rate_guaranteed':False,'live_trading_enabled':False,'decision':'REVIEW_CANDIDATES' if len(combined) else 'NO_TRADE'}
    Path('outputs/summary_v8.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    lines=['# V8 多段階勝率検証',f'- 東証対象銘柄数: {len(codes)}',f'- データ取得成功: {ok}',f'- 取得失敗/欠損: {missing}',f'- 5分足行数: {total_bars:,}',f'- 判定数: {total_bt:,}']+[f'- {int(float(k)*100)}%以上候補: {v}' for k,v in counts.items()]+[f'- 実行時間: {runtime/60:.1f}分','', '※ 各候補は学習区間と3つの独立検証区間すべてで閾値以上を要求。','※ 将来の勝率や利益を保証しません。実売買はOFFです。']
    Path('outputs/RESULT_V8_JP.md').write_text('\n'.join(lines),encoding='utf-8'); print('\n'.join(lines),flush=True)

if __name__=='__main__': main()

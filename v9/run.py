from __future__ import annotations
import argparse, importlib.util, json, math, time
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
V7=ROOT/'v7'/'run.py'
spec=importlib.util.spec_from_file_location('v7run', V7)
v7=importlib.util.module_from_spec(spec); assert spec.loader is not None; spec.loader.exec_module(v7)
GROUP=v7.GROUP


def bucket_daily_fee_safe(r):
    if r.empty: return pd.DataFrame()
    z=r.copy(); z['entry_date']=pd.to_datetime(z.entry_time).dt.date.astype(str)
    z['volume_bucket']=pd.cut(z.volume_ratio,[-1,1,1.5,2,3,5,10,float('inf')]).astype(str)
    z['vwap_bucket']=pd.cut(z.vwap_gap,[-float('inf'),-.02,-.01,-.005,0,.005,.01,.02,float('inf')]).astype(str)
    z['range_bucket']=pd.cut(z.range_pct,[-1,.0025,.005,.01,.02,.04,float('inf')]).astype(str)
    z['time_bucket']=pd.cut(z.minute_of_day,[0,555,570,600,660,690,750,810,870,930,1440],include_lowest=True).astype(str)
    z['net_positive']=(z.end_net_profit>0).astype(int)
    z['end_sq']=z.end_net_profit.astype(float)**2
    return (z.groupby(['entry_date']+GROUP,dropna=False,observed=True)
        .agg(samples=('end_net_profit','size'),positive=('net_positive','sum'),sum_end=('end_net_profit','sum'),sum_end_sq=('end_sq','sum'),sum_worst=('worst_net_profit','sum'),sum_best=('best_net_profit','sum')).reset_index())


def evaluate_fee_safe(daily,min_train=100,min_valid=30,max_avg_adverse=1200,folds=3,zscore=1.645):
    if daily.empty: return pd.DataFrame(),[]
    d=(daily.groupby(['entry_date']+GROUP,dropna=False,as_index=False)[['samples','positive','sum_end','sum_end_sq','sum_worst','sum_best']].sum())
    dates=sorted(d.entry_date.unique())
    if len(dates)<20: return pd.DataFrame(),[]
    base=max(10,int(len(dates)*.55)); fold_size=max(3,(len(dates)-base)//folds); merged=None; used=[]
    def summ(x,p):
        q=x.groupby(GROUP,dropna=False,as_index=False)[['samples','positive','sum_end','sum_end_sq','sum_worst','sum_best']].sum()
        n=q.samples.astype(float); mean=q.sum_end/n
        var=((q.sum_end_sq-(q.sum_end*q.sum_end)/n)/(n-1).clip(lower=1)).clip(lower=0)
        se=(var/n).pow(.5); lb=mean-zscore*se
        q[p+'_samples']=q.samples; q[p+'_positive_rate']=q.positive/n; q[p+'_avg_end']=mean; q[p+'_lb95_end']=lb
        q[p+'_avg_worst']=q.sum_worst/n; q[p+'_avg_best']=q.sum_best/n
        return q[GROUP+[p+'_samples',p+'_positive_rate',p+'_avg_end',p+'_lb95_end',p+'_avg_worst',p+'_avg_best']]
    for k in range(folds):
        te=base+k*fold_size; ve=len(dates) if k==folds-1 else min(len(dates),te+fold_size)
        if ve<=te: continue
        tr=d[d.entry_date.isin(dates[:te])]; va=d[d.entry_date.isin(dates[te:ve])]
        a=summ(tr,f'f{k}_train'); b=summ(va,f'f{k}_valid'); m=a.merge(b,on=GROUP,how='inner')
        m=m[(m[f'f{k}_train_samples']>=min_train)&(m[f'f{k}_valid_samples']>=min_valid)&
            (m[f'f{k}_train_avg_end']>0)&(m[f'f{k}_valid_avg_end']>0)&
            (m[f'f{k}_train_lb95_end']>0)&(m[f'f{k}_valid_lb95_end']>0)&
            (m[f'f{k}_valid_avg_worst']>=-abs(max_avg_adverse))]
        merged=m if merged is None else merged.merge(m,on=GROUP,how='inner'); used.append((dates[0],dates[te-1],dates[te],dates[ve-1]))
    if merged is None or merged.empty: return pd.DataFrame(),used
    s=[c for c in merged if c.endswith('_valid_samples')]; e=[c for c in merged if c.endswith('_valid_avg_end')]; l=[c for c in merged if c.endswith('_valid_lb95_end')]; p=[c for c in merged if c.endswith('_valid_positive_rate')]; w=[c for c in merged if c.endswith('_valid_avg_worst')]
    merged['validation_samples_total']=merged[s].sum(axis=1); merged['validation_avg_end_min']=merged[e].min(axis=1); merged['validation_avg_end_mean']=merged[e].mean(axis=1); merged['validation_lb95_min']=merged[l].min(axis=1); merged['validation_positive_rate_min']=merged[p].min(axis=1); merged['validation_avg_worst_mean']=merged[w].mean(axis=1)
    merged['score']=merged['validation_lb95_min']*merged['validation_positive_rate_min']
    return merged.sort_values(['score','validation_samples_total'],ascending=[False,False]),used


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--self-test',action='store_true'); ap.add_argument('--period',default='60d'); ap.add_argument('--interval',default='5m'); ap.add_argument('--batch-size',type=int,default=20); ap.add_argument('--symbol-chunk',type=int,default=100); ap.add_argument('--shares',type=int,default=100); ap.add_argument('--slippage',type=float,default=1.0)
    a=ap.parse_args()
    if a.self_test:
        v7.self_test(); print('V9_SELF_TEST_OK'); return
    Path('outputs_v9').mkdir(exist_ok=True)
    uni=v7.build_universe(); uni.to_csv('outputs_v9/tse_symbols.csv',index=False); codes=uni.code.astype(str).tolist(); print('universe',len(codes),flush=True)
    reports=[]; dailies=[]; total_bars=0; total_bt=0; t0=time.time()
    for n,start in enumerate(range(0,len(codes),a.symbol_chunk),1):
        chunk=codes[start:start+a.symbol_chunk]; t=time.time(); bars,rep=v7.download(chunk,a.period,a.interval,a.batch_size); rep['chunk']=n; reports.append(rep); total_bars+=len(bars)
        r=v7.backtest_chunk(bars,a.shares,0,a.slippage); total_bt+=len(r); dailies.append(bucket_daily_fee_safe(r))
        print(f'chunk {n}: symbols={len(chunk)} bars={len(bars):,} tests={len(r):,} sec={time.time()-t:.1f}',flush=True)
    report=pd.concat(reports,ignore_index=True); report.to_csv('outputs_v9/download_report.csv',index=False)
    daily=pd.concat(dailies,ignore_index=True) if dailies else pd.DataFrame(); candidates,folds=evaluate_fee_safe(daily)
    candidates.to_csv('outputs_v9/fee_safe_candidates.csv',index=False)
    candidates.head(200).to_csv('outputs_v9/shortlist.csv',index=False)
    ok=int((report.status=='ok').sum()); missing=int((report.status!='ok').sum()); runtime=time.time()-t0
    summary={'engine':'V9_FEE_SAFE_EXPECTANCY','requested_period':a.period,'interval':a.interval,'universe_symbols':len(codes),'download_ok_symbols':ok,'download_missing_symbols':missing,'downloaded_bar_rows':int(total_bars),'backtest_rows':int(total_bt),'fee_safe_candidate_count':int(len(candidates)),'shares':a.shares,'slippage_yen_each_side_assumption':a.slippage,'fee_and_tax_model':'Monex per-trade fees + 20.315% tax on positive trade profit','selection_rule':'train and every validation fold: avg net > 0 and one-sided 95% lower confidence bound > 0; avg adverse >= -1200','runtime_seconds':round(runtime,1),'future_profit_guaranteed':False,'live_trading_enabled':False,'decision':'REVIEW_FEE_SAFE_CANDIDATES' if len(candidates) else 'NO_TRADE'}
    Path('outputs_v9/summary_v9.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    lines=['# V9 手数料負け回避・期待値検証',f'- 東証対象銘柄数: {len(codes)}',f'- データ取得成功: {ok}',f'- 取得失敗/欠損: {missing}',f'- 5分足行数: {total_bars:,}',f'- 5分/10分判定数: {total_bt:,}',f'- 手数料等控除後・期待値プラス候補: {len(candidates)}',f'- 実行時間: {runtime/60:.1f}分','', '条件: 売買手数料・20.315%税・売買各1円スリッページ控除後。学習区間と3独立検証区間すべてで平均損益>0、かつ95%信頼下限>0。','※ 将来利益は保証しません。実売買はOFFです。']
    Path('outputs_v9/RESULT_V9_JP.md').write_text('\n'.join(lines),encoding='utf-8'); print('\n'.join(lines),flush=True)

if __name__=='__main__': main()

"""Run V13 daily policy comparison from an enriched holdout CSV.
Research only; no orders. Future columns are evaluation labels only.
Required columns:
day,symbol,predicted_mfe,predicted_mae,upside_probability,crash_probability,
future_mfe,future_mae,future_end
"""
from pathlib import Path
import json
import pandas as pd
from strategy_compare import TradeEval, POLICIES, simulate_day, summarize
from risk_objective import Metrics, loss_first_score

IN=Path('v13/results/holdout_enriched.csv')
OUT=Path('v13/results/daily_compare')
OUT.mkdir(parents=True,exist_ok=True)
REQ={'day','symbol','predicted_mfe','predicted_mae','upside_probability','crash_probability','future_mfe','future_mae','future_end'}

def main():
    if not IN.exists():
        raise SystemExit('missing v13/results/holdout_enriched.csv; generate predictions first')
    df=pd.read_csv(IN)
    missing=REQ-set(df.columns)
    if missing: raise SystemExit(f'missing columns: {sorted(missing)}')
    all_daily=[]; ranking=[]
    for policy in POLICIES:
        daily=[]
        for day,g in df.groupby('day',sort=True):
            rows=[TradeEval(**{k:r[k] for k in TradeEval.__dataclass_fields__}) for _,r in g.iterrows()]
            x=simulate_day(rows,policy); x['day']=day; x['policy']=policy; daily.append(x)
        s=summarize(daily)
        profitable=sum(x['return']>0 for x in daily)/len(daily) if daily else 0
        m=Metrics(s.get('losing_day_rate',1),s.get('max_drawdown',1),s.get('worst_day',-1),s.get('mean_giveback',1),profitable,s.get('target6_rate',0),s.get('mean_day_return',0))
        s.update(policy=policy,profitable_day_rate=profitable,loss_first_score=loss_first_score(m))
        ranking.append(s); all_daily.extend(daily)
    pd.DataFrame(all_daily).to_csv(OUT/'daily_results.csv',index=False)
    rank=pd.DataFrame(ranking).sort_values(['loss_first_score','target6_rate'],ascending=False)
    rank.to_csv(OUT/'strategy_ranking.csv',index=False)
    best=rank.iloc[0].to_dict() if len(rank) else {}
    (OUT/'summary.json').write_text(json.dumps({'best':best,'all':ranking},ensure_ascii=False,indent=2,default=str),encoding='utf-8')
    print(rank.to_string(index=False))

if __name__=='__main__': main()

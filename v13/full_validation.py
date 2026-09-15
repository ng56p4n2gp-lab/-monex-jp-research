"""V13 research-only validation runner. No live orders."""
from pathlib import Path
import pandas as pd
from prediction_builder import build_walk_forward
from strategy_compare import TradeEval, POLICIES, simulate_day, summarize

SRC=Path('results_v13/learning_samples.csv')
OUT=Path('results_v13/full_validation')
OUT.mkdir(parents=True,exist_ok=True)

def main():
    d=pd.read_csv(SRC)
    for c in ('Datetime','datetime','timestamp','index'):
        if 'decision_at' not in d.columns and c in d.columns: d=d.rename(columns={c:'decision_at'})
    if 'decision_at' not in d.columns: raise ValueError('decision timestamp missing')
    p=build_walk_forward(d)
    p.to_csv(OUT/'holdout_enriched.csv',index=False)
    results=[]
    for policy in POLICIES:
        daily=[]
        for day,g in p.groupby('day',sort=True):
            rows=[TradeEval(str(day),str(r.symbol),float(r.predicted_mfe),float(r.predicted_mae),float(r.upside_probability),float(r.crash_probability),float(r.future_mfe),float(r.future_mae),float(r.future_end)) for _,r in g.iterrows()]
            x=simulate_day(rows,policy); x.update(day=str(day),policy=policy); daily.append(x)
        s=summarize(daily); s['policy']=policy; results.append(s)
    rank=pd.DataFrame(results).sort_values(['losing_day_rate','max_drawdown','target6_rate'],ascending=[True,True,False])
    rank.to_csv(OUT/'strategy_ranking.csv',index=False)
    print(rank.to_string(index=False))

if __name__=='__main__': main()

"""Exploratory threshold curve, AFTER the primary experiment. Not a fresh test.

Run after b8802b_feature_experiment.py --score-design levels with its documented
output directory. Uses the calibration quantiles already recorded, no refit.
"""
import pickle
from pathlib import Path

import pandas as pd
import b8802b_feature_experiment as e


def main():
    p = e.ROOT / "results/b8802b_feature_candidates_levels"
    c = e.Candidates()
    with (p / "candidates.pkl").open("rb") as f:
        c.__dict__.update(pickle.load(f))
    raw, old = e.load_data()
    pre, po = e.prepare(raw), e.prepare(old)
    real = e.si._instantes_com_operacao(e.BUNDLE, raw, pre.index)
    realo = e.si._instantes_com_operacao(e.BUNDLE, old, po.index)
    syns = {}
    for date in ["2026-06-10", "2026-07-15"]:
        t0 = pd.Timestamp(date)
        rw = raw.loc[t0-pd.Timedelta(days=2):t0+pd.Timedelta(days=4)]
        pp = e.prepare(rw)
        rr = e.si._instantes_com_operacao(e.BUNDLE, rw, pp.index)
        for scale in [0, .5, 1]:
            x, tf, end = e.synthetic(pp, t0, 50, scale)
            syns[(date, scale)] = (c.predict(x.loc[rr]), rr, t0, tf, end)
    rows = []
    cal = pd.read_csv(p / "calibration.csv")
    for name in e.NAMES[1:]:
        current = pd.read_parquet(p / f"{name}_scores.parquet").score
        historical = pd.read_parquet(p / f"{name}_2022_scores.parquet").score
        for row in cal[(cal.model == name) & cal.alert_points.eq(0)].itertuples():
            out = e.alarms(current, row.threshold, real)
            w = out.loc["2026-01-01":]
            h = e.alarms(historical, row.threshold, realo)
            pf = h.loc["2022-06-29":"2022-07-06 09:59"]
            first = pf.index[pf.alert].min() if pf.alert.any() else None
            r = dict(model=name, q=row.quantile, threshold=row.threshold, selected_original=row.selected,
                episodes_2026=e.episode_count(w.alert), alert_hours_2026=float(w.alert.sum()/12),
                anchor=bool(out.loc["2026-01-17 19:30":"2026-01-18 00:40"].alert.any()),
                historical_lead_h=(e.FAILURE-first).total_seconds()/3600 if first is not None else None)
            for (date, scale), (s, rr, t0, tf, end) in syns.items():
                a = e.alarms(s[name].reindex(rr.index), row.threshold, rr).loc[t0:end]
                fi = a.index[a.alert].min() if a.alert.any() else None
                r[f"{date}_{scale}_lead_h"] = (tf-fi).total_seconds()/3600 if fi is not None else None
            rows.append(r)
            if name == "C1_vibracao" and row.quantile == 1:
                out.to_parquet(p / "C1_conservative_exploratory_scores.parquet")
                h.to_parquet(p / "C1_conservative_exploratory_2022_scores.parquet")
                flags = w.index[w.alert].to_series()
                groups = flags.diff().gt(pd.Timedelta("1h")).cumsum()
                episodes = []
                for _, ix in flags.groupby(groups):
                    observed = pre.loc[ix]
                    episodes.append({"start": str(ix.min()), "end": str(ix.max()),
                        "alert_points": len(ix), "alert_hours": len(ix)/12,
                        **{f"median_{col}": observed[col].median() for col in pre.columns}})
                pd.DataFrame(episodes).to_csv(p / "C1_conservative_exploratory_episodes.csv", index=False)
    result = pd.DataFrame(rows)
    result.to_csv(p / "threshold_tradeoff_exploratory.csv", index=False)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()

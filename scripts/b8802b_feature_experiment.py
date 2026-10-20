"""Offline, chronological B8802B feature experiment; no ClearML or deploy writes.

Run: OMP_NUM_THREADS=2 .venv/bin/python scripts/b8802b_feature_experiment.py
Candidate definitions and calibration policy are fixed before evaluation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "deploy_v2/Transpetro"))
import simpred_inference as si

BUNDLE = ROOT / "deploy_v2/Transpetro/B-8802B-2025/modelos/model_2025-01-01_2026-08-10_VAE"
VIB = ["Vibração Bomba LA", "Vibração Bomba LNA"]
PRESS = ["Pressão Sucção", "Pressão Descarga"]
TEMP = "Temperatura Bomba LA"
NAMES = ["VAE_atual", "C1_vibracao", "C2_pressao", "C3_dinamica_termica"]
LABELS = ["VAE atual", "C1: vibração", "C2: + pressões", "C3: + dinâmica térmica"]
FIT_END = pd.Timestamp("2025-10-01")
CAL_END = pd.Timestamp("2026-01-01")
FAILURE = pd.Timestamp("2022-07-06 10:00")
QUANTILES = [.95, .975, .99, .995, .9975, .999, .9995, .9999, 1.0]
POLICY = {
    "fit": ["2025-01-01", "2025-10-01 (exclusive)"],
    "calibration": ["2025-10-01", "2026-01-01 (exclusive)"],
    "evaluation": ["2026-01-01", "2026-08-10"],
    "calibration_objective": "lowest fixed quantile threshold with zero operational alerts in calibration",
    "quantiles": QUANTILES,
    "alarm": "15 of 20 valid 5-minute readings in 100min, then deploy minimum duration of 1h",
    "episode_gap": "1h (production convention)",
    "synthetic_coverage_min": .8,
    "models": "deterministic quantile boosting, max_iter=150, max_leaf_nodes=15, min_samples_leaf=150, l2=10, learning_rate=.05, early_stopping=False",
    "candidate_score": "positive maximum of robustly standardized 30min median levels and 1h-minus-6h trends; C2/C3 also include the C1 physical level/trend channels",
    "target": "current vibration; never current/lagged vibration as regression inputs",
    "temperature": "C3 uses only dT30min, dT2h and T-minus-causal-6h-median; C1/C2 use no T",
    "clipping": "baseline uses production clipping; candidates retain physical vibration amplitude (no learned upper clipping)",
    "scope": "research comparison, no production replacement; recent intervals are not fully labelled",
}


def load_data():
    recent = pd.read_feather(ROOT / "Dados-novos/B-8802B-2025.feather").set_index("datetime").sort_index()
    old = pd.read_feather(ROOT / "Dados/B-8802B.feather").sort_index()
    return recent, old


def prepare(raw):
    df = raw.copy()
    for s in json.loads((BUNDLE / "pipeline.json").read_text()):
        if s["step"] not in ("clip", "normalize"):
            df = si._STEPS_TEMPORAIS[s["step"]](df, **{k: v for k, v in s.items() if k != "step"})
    return df


def continuous_groups(index):
    return index.to_series().diff().ne(pd.Timedelta("5min")).cumsum()


def rolling(df, periods, min_periods=1):
    """All rows are 5min within a group; reset at every missing/filtered interval."""
    g = continuous_groups(df.index)
    return df.groupby(g.to_numpy()).rolling(periods, min_periods=min_periods).median().droplevel(0)


def lag(df, periods):
    return df.groupby(continuous_groups(df.index).to_numpy()).shift(periods)


def context(df, thermal=False):
    p = df[PRESS]
    pieces = [p, rolling(p, 6).add_suffix("_med30m"), rolling(p, 24, 6).add_suffix("_med2h"),
              (p-lag(p, 6)).add_suffix("_delta30m")]
    if thermal:
        t = rolling(df[[TEMP]].round(3), 3)
        # Sub-ulp differences around zero can cross tree split boundaries after
        # a constant offset. Quantize well below the sensor resolution so the
        # intended translation invariance also holds numerically.
        pieces += [(t-lag(t, 6)).round(3).rename(columns={TEMP: "dT30m"}),
                   (t-lag(t, 24)).round(3).rename(columns={TEMP: "dT2h"}),
                   (t-rolling(t, 72, 36)).round(3).rename(columns={TEMP: "T_rel6h"})]
    return pd.concat(pieces, axis=1)


def channels(v):
    return pd.concat([rolling(v, 6).add_suffix("_level30m"),
                      (rolling(v, 12, 6)-rolling(v, 72, 36)).add_suffix("_trend1h6h")], axis=1)


def robust_fit(frame):
    center = frame.median()
    # MAD with an IQR fallback and strictly positive floor for almost constant channels.
    mad = (frame-center).abs().median()*1.4826
    iqr = (frame.quantile(.75)-frame.quantile(.25))/1.349
    scale = pd.concat([mad, iqr], axis=1).max(axis=1).clip(lower=.01)
    return center, scale


def standardized(frame, norm):
    return ((frame-norm[0])/norm[1]).clip(lower=0).max(axis=1)


def alarms(score, threshold, real):
    above = (score > threshold)[real].astype(float)
    a = ((above.rolling("100min").sum() >= 15) & (above.rolling("100min").count() >= 20))
    a = a.reindex(score.index, fill_value=False)
    return pd.DataFrame({"score": score, "alarm": a, "alert": si._duracao_minima(a, 1.0)})


def episode_count(flag):
    ix = flag[flag].index.to_series()
    return 0 if ix.empty else int(ix.diff().gt(pd.Timedelta("1h")).sum()+1)


def summarize(res, real):
    f = res.alert
    hours = float(real.sum()/12)
    return {"n": len(res), "operating_hours": hours, "alert_points": int(f.sum()),
            "alert_pct_valid_operating": float(100*f.sum()/max(1, real.sum())),
            "episodes": episode_count(f), "episodes_per_1000h": episode_count(f)/max(hours, 1e-9)*1000,
            "first_alert": str(f[f].index.min()) if f.any() else None}


class Candidates:
    def __init__(self, score_design="levels_trends"):
        self.score_design = score_design

    def score_channels(self, df):
        c = channels(df)
        return c.filter(like="_level30m") if self.score_design == "levels" else c

    def fit(self, df):
        self.base_norm = robust_fit(self.score_channels(df[VIB]))
        self.regressors = {}
        self.residual_norm = {}
        for name, thermal in [(NAMES[2], False), (NAMES[3], True)]:
            x = context(df, thermal)
            models = []
            prediction = pd.DataFrame(index=df.index)
            for col in VIB:
                m = HistGradientBoostingRegressor(loss="quantile", quantile=.5, learning_rate=.05,
                    max_iter=150, max_leaf_nodes=15, min_samples_leaf=150, l2_regularization=10,
                    early_stopping=False, random_state=0)
                m.fit(x, df[col])
                prediction[col] = m.predict(x)
                models.append(m)
            self.regressors[name] = models
            self.residual_norm[name] = robust_fit(self.score_channels(df[VIB]-prediction))
            print(f"Ajustado {name}: {len(df)} pontos, {len(x.columns)} preditores", flush=True)
        return self

    def predict(self, df):
        base = standardized(self.score_channels(df[VIB]), self.base_norm)
        result = {NAMES[1]: base}
        for name, thermal in [(NAMES[2], False), (NAMES[3], True)]:
            x = context(df, thermal)
            pred = pd.DataFrame({c: m.predict(x) for c, m in zip(VIB, self.regressors[name])}, index=df.index)
            residual = standardized(self.score_channels(df[VIB]-pred), self.residual_norm[name])
            result[name] = pd.concat([base, residual], axis=1).max(axis=1)
        return result


class Baseline:
    def __init__(self):
        self.model = si.carregar_modelo(BUNDLE)
        self.threshold = json.loads((BUNDLE / "alarm.json").read_text())["threshold"]
        self.bounds = json.loads((BUNDLE / "clip_bounds.json").read_text())
        with (BUNDLE / "scaler.pkl").open("rb") as f:
            self.scaler = pickle.load(f)

    def predict(self, pre):
        df = pre.copy()
        for c, (lo, hi) in self.bounds.items():
            df[c] = df[c].clip(lo, hi)
        with torch.no_grad():
            x = torch.tensor(self.scaler.transform(df.values), dtype=torch.float32)
            recon, _ = self.model(x)
            score = ((recon-x)**2).mean(1).numpy()
        return pd.Series(score, index=df.index)


def calibrate(scores, real):
    thresholds, rows = {}, []
    for name, s in scores.items():
        if name == NAMES[0]:
            continue
        chosen = False
        cal = s[(s.index >= FIT_END) & (s.index < CAL_END)]
        r = real.reindex(cal.index)
        for q in QUANTILES:
            t = float(cal[r].quantile(q))
            a = alarms(cal, t, r)
            passed = not a.alert.any()
            selected = passed and not chosen
            rows.append({"model": name, "quantile": q, "threshold": t,
                         **summarize(a, r), "selected": selected})
            if selected:
                thresholds[name] = t
                chosen = True
    return thresholds, pd.DataFrame(rows)


def check_features(df):
    sample = df.iloc[:2400]
    full = context(sample, True)
    pd.testing.assert_frame_equal(full.iloc[:1200], context(sample.iloc[:1200], True))
    shifted = sample.copy()
    shifted[TEMP] += 10
    np.testing.assert_allclose(full, context(shifted, True), atol=1e-10, equal_nan=True)
    # Gaps must not be bridged by a lag or smoothing history.
    gap = sample.iloc[np.r_[0:100, 200:300]]
    assert lag(gap[[TEMP]], 6).iloc[100:106].isna().all().all()
    pd.testing.assert_series_equal(rolling(gap[[TEMP]], 6).iloc[100], gap[[TEMP]].iloc[100])


def synthetic(pre, t0, temp, scale, shape="ramp48", thermal_only=False):
    df = pre.copy()
    ramp_h = 168 if shape == "ramp168" else (0 if shape == "step" else 48)
    total_h = ramp_h+24
    dt = np.asarray((df.index-t0).total_seconds())/3600
    prof = np.where((dt >= 0) & (dt <= total_h), np.clip(dt/max(ramp_h, 1e-9), 0, 1), 0)
    if shape == "step":
        prof = ((dt >= 0) & (dt <= total_h)).astype(float)
    df[TEMP] = float(temp)
    if thermal_only:
        df[TEMP] += scale*10*prof
    else:
        for col, amp in zip(VIB, [1.515, 1.237]):
            df[col] += scale*amp*prof
    return df, t0+pd.Timedelta(hours=ramp_h), t0+pd.Timedelta(hours=total_h)


def evaluate_synthetic(raw, cand, baseline, thresholds):
    rows = []
    specs = [(d, "ramp48", False) for d in ["2026-01-25", "2026-03-05", "2026-04-20", "2026-06-10", "2026-07-15"]]
    specs += [(d, shape, False) for d in ["2026-06-10", "2026-07-15"] for shape in ["step", "ramp168"]]
    specs += [("2026-06-10", "ramp48", True), ("2026-06-10", "step", True)]
    for date, shape, thermal_only in specs:
        t0 = pd.Timestamp(date)
        raw_window = raw.loc[t0-pd.Timedelta(days=2):t0+pd.Timedelta(days=9)]
        pre = prepare(raw_window)
        real = si._instantes_com_operacao(BUNDLE, raw_window, pre.index)
        # Candidate time features reset at all missing and nonoperating intervals.
        for temp in [40, 50, 60]:
            for scale in ([0, 1] if thermal_only else [0, .25, .5, 1]):
                df, tf, end = synthetic(pre, t0, temp, scale, shape, thermal_only)
                scores = cand.predict(df.loc[real])
                scores[NAMES[0]] = baseline.predict(df)
                valid = real.loc[t0:end]
                grid_n = int((end-t0)/pd.Timedelta("5min"))+1
                cover = float(valid.sum()/grid_n)
                for name, s in scores.items():
                    s = s.reindex(pre.index)
                    a = alarms(s, thresholds[name], real).loc[t0:end]
                    first = a.index[a.alert].min() if a.alert.any() else None
                    rows.append({"model": name, "date": date, "shape": shape, "thermal_only": thermal_only,
                        "temperature": temp, "scale": scale, "coverage": cover,
                        "n_valid": int(valid.sum()), "first_alert": str(first) if first is not None else None,
                        "lead_hours": (tf-first).total_seconds()/3600 if first is not None else np.nan})
        print(f"Sondagem {date} {shape} thermal_only={thermal_only}", flush=True)
    result = pd.DataFrame(rows)
    keys = ["model", "date", "shape", "thermal_only", "temperature"]
    controls = result[result.scale.eq(0)].set_index(keys).first_alert.notna()
    result["baseline_contaminated"] = [bool(controls.loc[tuple(getattr(r, k) for k in keys)]) for r in result.itertuples()]
    result["eligible"] = (result.coverage >= .8) & ~result.baseline_contaminated & result.scale.gt(0)
    return result


def summarize_injections(df):
    rows = []
    # Common eligibility across models: no method is rewarded for contaminating a difficult window.
    keys = ["date", "shape", "thermal_only", "temperature", "scale"]
    common = df.groupby(keys).eligible.all()
    df = df.copy()
    df["common_eligible"] = [bool(common.loc[tuple(getattr(r, k) for k in keys)]) for r in df.itertuples()]
    for (model, shape, thermal, scale), g in df[df.scale.gt(0)].groupby(["model", "shape", "thermal_only", "scale"]):
        w = g[g.common_eligible]
        found = w.lead_hours.notna()
        before = found & w.lead_hours.ge(0)
        rows.append({"model": model, "shape": shape, "thermal_only": thermal, "scale": scale,
            "eligible": len(w), "detected": int(found.sum()), "before_ramp_end": int(before.sum()),
            "median_lead_detected_h": w.lead_hours.median(), "worst_lead_detected_h": w.lead_hours.min(),
            "missed": int((~found).sum())})
    return df, pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="results/b8802b_feature_candidates")
    ap.add_argument("--score-design", choices=["levels_trends", "levels"], default="levels_trends")
    args = ap.parse_args()
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    threadpool_limits(limits=2)
    policy = dict(POLICY, score_design=args.score_design)
    if args.score_design == "levels":
        policy["candidate_score"] = "ablation: positive maximum of robustly standardized 30min median levels; remove trend channels; preserve C1 backup in C2/C3"
    (out/"protocol.json").write_text(json.dumps(policy, ensure_ascii=False, indent=2))
    raw, old = load_data()
    pre, pre_old = prepare(raw), prepare(old)
    real = si._instantes_com_operacao(BUNDLE, raw, pre.index)
    real_old = si._instantes_com_operacao(BUNDLE, old, pre_old.index)
    physical = pre.loc[real]
    check_features(physical)
    print("Features causais, invariância aditiva e reset de gaps: OK", flush=True)
    fit = physical.loc[physical.index < FIT_END]
    cand = Candidates(args.score_design).fit(fit)
    baseline = Baseline()
    scores = {k: s.reindex(pre.index) for k, s in cand.predict(physical).items()}
    scores[NAMES[0]] = baseline.predict(pre)
    thresholds, cal = calibrate(scores, real)
    thresholds[NAMES[0]] = baseline.threshold
    cal.to_csv(out/"calibration.csv", index=False)
    (out/"thresholds.json").write_text(json.dumps(thresholds, indent=2))
    with (out/"candidates.pkl").open("wb") as f:
        pickle.dump(vars(cand), f)
    print("Limiares congelados:", thresholds, flush=True)

    results = {name: alarms(s, thresholds[name], real) for name, s in scores.items()}
    reference = si.prever(BUNDLE, baseline.model, si.preprocessar(BUNDLE, raw), df_bruto=raw)
    np.testing.assert_allclose(results[NAMES[0]].score, reference.reconstruction_error, rtol=1e-6)
    assert np.array_equal(results[NAMES[0]].alert, reference.alerta)
    assert np.array_equal(results[NAMES[0]].alarm, reference.is_anomaly)
    print("Score e persistência equivalentes ao deploy: OK", flush=True)
    old_scores = {k: s.reindex(pre_old.index) for k, s in cand.predict(pre_old.loc[real_old]).items()}
    old_scores[NAMES[0]] = baseline.predict(pre_old)
    old_results = {name: alarms(s, thresholds[name], real_old) for name, s in old_scores.items()}
    rows, event_rows, monthly = [], [], []
    periods = {"fit2025": ("2025-01-01", str(FIT_END)), "cal2025": (str(FIT_END), str(CAL_END)),
        "jan-mai2026": (str(CAL_END), "2026-06-01"), "jun-ago2026": ("2026-06-01", "2026-08-11")}
    for name in NAMES:
        r, ro = results[name], old_results[name]
        for period, (a, b) in periods.items():
            w = r[(r.index >= a) & (r.index < b)]
            rows.append({"model": name, "period": period, **summarize(w, real.reindex(w.index))})
        normal = ro[ro.index < "2022-06-29"]
        rows.append({"model": name, "period": "normal2022", **summarize(normal, real_old.reindex(normal.index))})
        pf = ro[(ro.index >= "2022-06-29") & (ro.index < FAILURE)]
        first = pf.index[pf.alert].min() if pf.alert.any() else None
        anchor = r.loc["2026-01-17 19:30":"2026-01-18 00:40"].alert.any()
        event_rows.append({"model": name, "historical_lead_h": (FAILURE-first).total_seconds()/3600 if first is not None else np.nan,
            "historical_first_alert": str(first), "prefailure_alert_pct": 100*pf.alert.mean(), "january_anchor_detected": bool(anchor)})
        for date, w in r.groupby(pd.Grouper(freq="MS")):
            if len(w): monthly.append({"model": name, "month": str(date.date()), **summarize(w, real.reindex(w.index))})
        r.to_parquet(out/f"{name}_scores.parquet")
        ro.to_parquet(out/f"{name}_2022_scores.parquet")
    history, events, monthly = pd.DataFrame(rows), pd.DataFrame(event_rows), pd.DataFrame(monthly)
    history.to_csv(out/"history.csv", index=False)
    events.to_csv(out/"events.csv", index=False)
    monthly.to_csv(out/"monthly.csv", index=False)
    print("HISTÓRICO\n", history.to_string(index=False), "\nEVENTOS\n", events.to_string(index=False), flush=True)

    counter = []
    # Exact offset invariance preserves observed thermal dynamics; constant levels remove it.
    for kind, values in [("constant", [40, 50, 60]), ("offset", [-10, 0, 10])]:
        ref = None
        for val in values:
            changed = pre.copy()
            changed[TEMP] = val if kind == "constant" else pre[TEMP]+val
            cs = {k: s.reindex(pre.index) for k, s in cand.predict(changed.loc[real]).items()}
            cs[NAMES[0]] = baseline.predict(changed)
            if ref is None: ref = cs
            for name, s in cs.items():
                delta = float((s-ref[name]).abs().max())
                if name != NAMES[0]: assert delta < 1e-8, (name, kind, val, delta)
                rr = alarms(s, thresholds[name], real)
                for period, (a, b) in list(periods.items())[2:]:
                    w = rr[(rr.index >= a) & (rr.index < b)]
                    counter.append({"model": name, "intervention": kind, "value": val, "period": period,
                        "max_score_change_vs_first_value": delta, **summarize(w, real.reindex(w.index))})
    pd.DataFrame(counter).to_csv(out/"temperature_counterfactual.csv", index=False)
    syn, sy_summary = summarize_injections(evaluate_synthetic(raw, cand, baseline, thresholds))
    syn.to_csv(out/"synthetic.csv", index=False)
    sy_summary.to_csv(out/"synthetic_summary.csv", index=False)
    print("SINTÉTICAS\n", sy_summary.to_string(index=False), flush=True)
    # Training artifacts round-trip on a small, independent evaluation slice.
    restored = Candidates()
    with (out/"candidates.pkl").open("rb") as f: restored.__dict__.update(pickle.load(f))
    sl = physical.loc["2026-06-01":"2026-06-03"]
    for name, score in cand.predict(sl).items():
        np.testing.assert_array_equal(score, restored.predict(sl)[name])
    metadata = {"checks": ["feature prefix causality", "gap resets", "additive thermal invariance", "deploy score/alarm/alert parity",
                           "all candidate constant-level and offset invariance", "model serialization round-trip"],
        "n_fit": len(fit), "n_calibration": int(((physical.index>=FIT_END)&(physical.index<CAL_END)).sum()),
        "n_recent_processed": len(pre), "n_recent_real": int(real.sum()),
        "bundle": str(BUNDLE.relative_to(ROOT)),
        "baseline_weights_sha256": hashlib.sha256((BUNDLE/"model_state.pt").read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (out/"checks.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
    draw(out, history, events, monthly, syn, old_results, thresholds)
    print(f"Concluído: {out}", flush=True)


def draw(out, history, events, monthly, syn, old_results, thresholds):
    colors = ["#64748b", "#0284c7", "#059669", "#d97706"]
    fig, ax = plt.subplots(2, 2, figsize=(13, 8))
    for name, label, color in zip(NAMES, LABELS, colors):
        m = monthly[(monthly.model==name)&(monthly.month >= "2026-01")]
        ax[0, 0].plot(pd.to_datetime(m.month), m.episodes, "o-", label=label, color=color)
        r = old_results[name].loc["2022-07-01":"2022-07-06 10:00"]
        ax[0, 1].plot(r.index, (r.score/thresholds[name]).clip(upper=6), label=label, color=color, alpha=.8)
        s = syn[(syn.model==name)&(syn.date=="2026-06-10")&(syn["shape"]=="ramp48")&~syn.thermal_only&syn.scale.eq(1)]
        ax[1, 0].plot(s.temperature, s.lead_hours, "o-", label=label, color=color)
        q = syn[(syn.model==name)&(syn.date=="2026-06-10")&(syn["shape"]=="ramp48")&~syn.thermal_only&syn.scale.eq(.5)]
        ax[1, 1].plot(q.temperature, q.lead_hours, "o-", label=label, color=color)
    ax[0, 0].set(title="Episódios de alerta em 2026 (sem rótulos completos)", ylabel="Episódios por mês")
    ax[0, 1].axhline(1, color="black", linestyle="--", linewidth=1)
    ax[0, 1].set(title="Falha real: score / limiar (visual limitado a 6)", ylabel="Score relativo")
    for a, title in zip(ax[1], ["Rampa de vibração a 100% — junho/2026", "Rampa de vibração a 50% — junho/2026"]):
        a.axhline(0, color="black", linewidth=.8)
        a.set(title=title, xlabel="Temperatura imposta (°C)", ylabel="Antecedência ao fim da rampa (h)", xticks=[40, 50, 60])
    ax[0, 0].legend(fontsize=8)
    for a in ax.flat: a.grid(alpha=.2)
    fig.autofmt_xdate(rotation=20)
    fig.suptitle("B8802B — comparação exploratória com persistência de produção", fontsize=14)
    fig.tight_layout()
    fig.savefig(out/"comparison.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    main()

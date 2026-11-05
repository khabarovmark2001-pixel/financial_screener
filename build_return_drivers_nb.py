import json
from pathlib import Path

BASE = Path("/Users/markkhabarov/Desktop/financial market/Portfolio screener")

cells = []


def _lines(text):
    """nbformat wants each source line to keep its newline, bar the last."""
    rows = text.strip("\n").split("\n")
    return [row + "\n" for row in rows[:-1]] + rows[-1:]


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {},
                  "source": _lines(text)})


def code(text):
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {},
                  "outputs": [], "source": _lines(text)})


md(r'''
# What explains forward returns?

Evaluating `data/valuation_history.parquet` — 61,508 point-in-time observations,
2,617 tickers, 2019-09 to 2026-09 — as a factor study.

**Question.** Given what we knew about a company on its filing date, how much of
its subsequent return can we explain?

**Two families of determinant.**

| | Variables |
|---|---|
| **Local** (firm-level) | DCF outputs by model, blended fair value, multiples, quality and margin ratios, engine meta (confidence, model agreement) |
| **Global** (grouping) | sector, industry, company size |

**Target.** `forward_return_{1m, 3m, 6m, 12m}` — total return from the filing date.

**Method.** Everything is measured *cross-sectionally within each quarter*, so
this asks "which stocks beat other stocks" rather than "was the market up". The
headline test is a Fama–MacBeth regression with Newey–West standard errors,
because forward returns from consecutive quarterly filings overlap heavily and
ordinary standard errors on them are badly overstated.
''')

code(r'''
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

import valuation_history as vh

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 60)
plt.rcParams.update({"figure.figsize": (11, 4.2), "axes.grid": True,
                     "grid.alpha": 0.25, "axes.spines.top": False,
                     "axes.spines.right": False, "font.size": 10})

ACCENT, MUTED, UP, DOWN = "#2F5D8A", "#8894A0", "#1B7F5E", "#B4472F"

HORIZONS = ["1m", "3m", "6m", "12m"]
# Forward windows in months, used to set the Newey-West lag: consecutive
# quarterly observations of a 12-month return share three quarters of their
# path, so residuals are autocorrelated out to that many lags.
OVERLAP_QUARTERS = {"1m": 1, "3m": 1, "6m": 2, "12m": 4}

panel = vh.load()
panel["quarter"] = pd.PeriodIndex(panel["quarter"], freq="Q")
print(f"{len(panel):,} rows | {panel['ticker'].nunique():,} tickers | "
      f"{panel['quarter'].nunique()} quarters | "
      f"{panel['date'].min().date()} to {panel['date'].max().date()}")
panel[["ticker", "date", "sector", "fair_value_upside", "forward_return_12m"]].head()
''')

md(r'''
## 1. Features

**Local determinants** fall into four blocks. Valuation signals are expressed as
*upside* — model value over market price — rather than as a level, because a
$400 fair value means nothing without knowing the price beside it.

Multiples are **inverted into yields** (E/P rather than P/E). A price multiple is
unbounded above and discontinuous through zero; its reciprocal is bounded,
monotone in cheapness, and does not explode when earnings approach zero.
''')

code(r'''
def safe_ratio(numerator, denominator):
    """Ratio that returns NaN rather than infinity on a zero denominator."""
    denominator = pd.to_numeric(denominator, errors="coerce").replace(0, np.nan)
    return pd.to_numeric(numerator, errors="coerce") / denominator


price = panel["market_price"]

# --- local: valuation signals -------------------------------------------
panel["upside_blended"] = panel["fair_value_upside"]
panel["upside_multiples"] = safe_ratio(panel["fair_value_multiples"], price) - 1
for model in ["fcfe", "fcff", "ddm", "rim", "relative"]:
    panel[f"upside_{model}"] = safe_ratio(panel[f"{model}_value"], price) - 1

# --- local: multiples, as yields ----------------------------------------
panel["earnings_yield"] = safe_ratio(1.0, panel["pe_ttm"])
panel["sales_yield"] = safe_ratio(1.0, panel["ps_ttm"])
panel["book_yield"] = safe_ratio(1.0, panel["pb"])
panel["ebitda_yield"] = safe_ratio(1.0, panel["ev_ebitda_ttm"])

# --- local: quality and profitability ------------------------------------
QUALITY = ["roe_ttm", "roic_ttm", "ebit_margin_ttm", "net_margin_ttm"]

# --- local: how much the engine trusted itself ---------------------------
ENGINE_META = ["confidence", "weighted_reliability", "models_used",
               "model_agreement"]

# NOTE: `fair_value_multiples` and `relative_value` are the same number - the
# engine assigns the peer-multiple blend to both. They are kept separate here
# only so the IC table shows them agreeing exactly, which is a useful check
# that this notebook is reading the columns it thinks it is. Never put both in
# one regression: it is the same variable twice.
VALUATION = ["upside_blended", "upside_multiples", "upside_fcfe", "upside_fcff",
             "upside_ddm", "upside_rim", "upside_relative"]
MULTIPLES = ["earnings_yield", "sales_yield", "book_yield", "ebitda_yield"]
LOCAL = VALUATION + MULTIPLES + QUALITY + ENGINE_META

# --- global: sector, industry, size --------------------------------------
panel["log_size"] = np.log(panel["marketcap"].where(panel["marketcap"] > 0))
GLOBAL_CATEGORICAL = ["sector", "industry"]
GLOBAL_NUMERIC = ["log_size"]

coverage = panel[LOCAL + GLOBAL_NUMERIC].notna().mean().sort_values()
print("non-null coverage:")
print(coverage.to_string(float_format=lambda x: f"{x:.1%}"))
''')

md(r'''
### Cross-sectional ranking

Every feature is converted to its **percentile rank within its own quarter**,
centred on zero. This does three things at once:

- removes the market-wide move, so a coefficient measures stock selection
- makes a coefficient directly readable as the return spread between the best
  and worst stock on that feature
- destroys outliers, which matters here — decile 9 of `upside_blended` has a
  median upside of +300%, and on raw values a handful of those would dominate
  any regression
''')

code(r'''
def cross_sectional_rank(frame, columns, by="quarter", min_names=30):
    """Percentile rank within each period, centred on zero, range [-0.5, 0.5]."""
    out = pd.DataFrame(index=frame.index)
    sizes = frame.groupby(by)[by].transform("size")
    for column in columns:
        values = frame[column].where(sizes >= min_names)
        ranked = values.groupby(frame[by]).rank(pct=True)
        out[column] = ranked - 0.5
    return out


ranked = cross_sectional_rank(panel, LOCAL + GLOBAL_NUMERIC)
ranked = ranked.add_suffix("_rank")

data = pd.concat([panel[["ticker", "date", "quarter", "sector", "industry",
                         "marketcap", "profile"]
                        + [f"forward_return_{h}" for h in HORIZONS]],
                  ranked], axis=1)

RANKED_LOCAL = [f"{c}_rank" for c in LOCAL]
RANKED_GLOBAL = [f"{c}_rank" for c in GLOBAL_NUMERIC]

print(f"{len(data):,} rows, {len(RANKED_LOCAL)} local + {len(RANKED_GLOBAL)} "
      f"numeric global features")
data[RANKED_LOCAL].describe().T[["count", "mean", "std", "min", "max"]].head(8)
''')

md(r'''
## 2. Univariate: information coefficient

The **IC** is the cross-sectional Spearman correlation between a feature and the
forward return, computed separately in each quarter. Its *mean* says whether the
feature ranks stocks correctly on average; its *t-statistic across quarters* says
whether that is distinguishable from luck.

The t-statistic uses **Newey–West** standard errors with the lag set to the
overlap of the horizon. Ignoring that overlap would roughly double the apparent
significance of the 12-month results.
''')

code(r'''
def information_coefficient(frame, feature, horizon, min_names=30):
    """Per-quarter Spearman IC between a feature and the forward return."""
    target = f"forward_return_{horizon}"
    subset = frame[["quarter", feature, target]].dropna()
    per_quarter = subset.groupby("quarter").apply(
        lambda g: g[feature].corr(g[target], method="spearman")
        if len(g) >= min_names else np.nan, include_groups=False)
    return per_quarter.dropna()


def newey_west_t(series, lags):
    """Mean of a series with a Newey-West t-statistic, for overlapping data."""
    series = series.dropna()
    if len(series) < 4:
        return np.nan, np.nan
    model = sm.OLS(series.values, np.ones(len(series))).fit(
        cov_type="HAC", cov_kwds={"maxlags": lags})
    return float(model.params[0]), float(model.tvalues[0])


rows = []
for feature in RANKED_LOCAL + RANKED_GLOBAL:
    record = {"feature": feature.replace("_rank", "")}
    for horizon in HORIZONS:
        series = information_coefficient(data, feature, horizon)
        mean_ic, t_stat = newey_west_t(series, OVERLAP_QUARTERS[horizon])
        record[f"IC_{horizon}"] = mean_ic
        record[f"t_{horizon}"] = t_stat
    rows.append(record)

ic_table = pd.DataFrame(rows).set_index("feature")
ic_table = ic_table.sort_values("IC_12m", ascending=False)
display_columns = [c for h in HORIZONS for c in (f"IC_{h}", f"t_{h}")]
print("Information coefficient by feature and horizon (Newey-West t in brackets)")
ic_table[display_columns].style.format("{:+.3f}").background_gradient(
    cmap="RdYlGn", subset=[f"IC_{h}" for h in HORIZONS], vmin=-0.08, vmax=0.08)
''')

code(r'''
fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))

top = ic_table["IC_12m"].dropna().sort_values()
axes[0].barh(top.index, top.values,
             color=[UP if v > 0 else DOWN for v in top.values])
axes[0].axvline(0, color="black", lw=1)
axes[0].set_title("Mean 12-month IC by feature")
axes[0].set_xlabel("Spearman IC")

best = ic_table["IC_12m"].idxmax()
series = information_coefficient(data, f"{best}_rank", "12m")
axes[1].bar(series.index.astype(str), series.values,
            color=[UP if v > 0 else DOWN for v in series.values])
axes[1].axhline(series.mean(), color=ACCENT, ls="--",
                label=f"mean {series.mean():+.3f}")
axes[1].axhline(0, color="black", lw=1)
axes[1].set_title(f"12-month IC by quarter — {best}")
axes[1].tick_params(axis="x", rotation=90)
axes[1].legend()
plt.tight_layout()
plt.show()

hit_rate = (series > 0).mean()
print(f"{best}: positive IC in {hit_rate:.0%} of {len(series)} quarters")
''')

md(r'''
## 3. Decile spreads

The IC treats the whole cross-section linearly. Deciles ask a blunter question:
if you bought the top tenth on a signal and sold the bottom tenth, what did you
earn? The gap between them is the tradeable content — subject to every caveat in
§7.
''')

code(r'''
def decile_profile(frame, feature, horizon, bins=10):
    target = f"forward_return_{horizon}"
    subset = frame[["quarter", feature, target]].dropna().copy()
    subset["decile"] = subset.groupby("quarter")[feature].transform(
        lambda s: pd.qcut(s.rank(method="first"), bins, labels=False)
        if len(s) >= bins * 3 else np.nan)
    subset = subset.dropna(subset=["decile"])
    return subset.groupby("decile")[target].agg(["size", "mean", "median"])


headline = ["upside_blended_rank", "upside_multiples_rank", "earnings_yield_rank",
            "upside_rim_rank", "roic_ttm_rank", "log_size_rank"]

fig, axes = plt.subplots(2, 3, figsize=(14, 7))
for axis, feature in zip(axes.ravel(), headline):
    profile = decile_profile(data, feature, "12m")
    axis.bar(profile.index, profile["median"],
             color=[UP if v > 0 else DOWN for v in profile["median"]])
    spread = profile["median"].iloc[-1] - profile["median"].iloc[0]
    axis.set_title(f"{feature.replace('_rank','')}\nD9-D0 = {spread:+.1%}",
                   fontsize=10)
    axis.set_xlabel("decile")
    axis.yaxis.set_major_formatter(lambda v, p: f"{v:.0%}")
plt.suptitle("Median 12-month forward return by decile", y=1.02)
plt.tight_layout()
plt.show()
''')

md(r'''
## 4. Global determinants

Sector, industry and size are not signals you rank stocks on — they are the
groups within which ranking happens. Two questions matter:

1. **How much return variance do they explain on their own?**
2. **Does the local signal survive inside them**, or is it just a proxy for
   being in a cheap sector?
''')

code(r'''
target_column = "forward_return_12m"
sample = data.dropna(subset=[target_column, "upside_blended_rank"]).copy()

# --- how much does each grouping explain on its own ---------------------
def grouping_r2(frame, keys, target):
    """R^2 from predicting the target with group means alone."""
    fitted = frame.groupby(keys)[target].transform("mean")
    residual = frame[target] - fitted
    return 1.0 - residual.var() / frame[target].var()


explained = {
    "quarter (market timing)": grouping_r2(sample, ["quarter"], target_column),
    "sector": grouping_r2(sample, ["sector"], target_column),
    "industry": grouping_r2(sample, ["industry"], target_column),
    "quarter x sector": grouping_r2(sample, ["quarter", "sector"], target_column),
    "quarter x industry": grouping_r2(sample, ["quarter", "industry"], target_column),
}
print("Share of 12-month return variance explained by grouping alone")
for label, value in explained.items():
    print(f"  {label:<26} {value:6.1%}")
''')

code(r'''
# --- does the local signal work inside every sector? --------------------
by_sector = []
for sector, block in sample.groupby("sector"):
    series = information_coefficient(block, "upside_blended_rank", "12m",
                                     min_names=20)
    mean_ic, t_stat = newey_west_t(series, OVERLAP_QUARTERS["12m"])
    by_sector.append({"sector": sector, "n": len(block), "IC": mean_ic,
                      "t": t_stat, "quarters": len(series)})
sector_ic = pd.DataFrame(by_sector).set_index("sector").sort_values("IC")

# --- and at every size? --------------------------------------------------
sample["size_quintile"] = sample.groupby("quarter")["marketcap"].transform(
    lambda s: pd.qcut(s.rank(method="first"), 5, labels=
                      ["micro", "small", "mid", "large", "mega"]))
by_size = []
for bucket, block in sample.groupby("size_quintile", observed=True):
    series = information_coefficient(block, "upside_blended_rank", "12m",
                                     min_names=20)
    mean_ic, t_stat = newey_west_t(series, OVERLAP_QUARTERS["12m"])
    by_size.append({"size": bucket, "n": len(block), "IC": mean_ic, "t": t_stat})
size_ic = pd.DataFrame(by_size).set_index("size")

fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
axes[0].barh(sector_ic.index, sector_ic["IC"],
             color=[UP if v > 0 else DOWN for v in sector_ic["IC"]])
axes[0].axvline(0, color="black", lw=1)
axes[0].set_title("12-month IC of blended upside, by sector")
axes[1].bar(size_ic.index.astype(str), size_ic["IC"],
            color=[UP if v > 0 else DOWN for v in size_ic["IC"]])
axes[1].axhline(0, color="black", lw=1)
axes[1].set_title("12-month IC of blended upside, by size quintile")
plt.tight_layout()
plt.show()

print(sector_ic.round(3).to_string())
print()
print(size_ic.round(3).to_string())
''')

md(r'''
## 5. Fama–MacBeth regression

The headline multivariate test. In **each quarter** the forward return is
regressed cross-sectionally on the features; the coefficients from all quarters
are then averaged, and their t-statistic computed with Newey–West standard
errors at the horizon's overlap.

This is the standard treatment for exactly this data shape — a wide
cross-section observed over relatively few periods — and it sidesteps the
autocorrelation that overlapping forward returns induce in pooled OLS.

Three specifications separate local from global:

| | Regressors |
|---|---|
| **Local only** | valuation, multiples, quality, engine meta |
| **Local + global** | the above plus size, plus sector dummies |
| **Sector-neutral** | local features against a target demeaned within sector each quarter |
''')

code(r'''
def fama_macbeth(frame, features, target, lags, min_names=40,
                 sector_dummies=False):
    """Cross-sectional OLS each quarter, then average the coefficients."""
    columns = ["quarter", target] + features + (["sector"] if sector_dummies else [])
    subset = frame[columns].dropna()
    coefficients, r_squared = [], []

    for quarter, block in subset.groupby("quarter"):
        if len(block) < min_names:
            continue
        design = block[features].copy()
        if sector_dummies:
            dummies = pd.get_dummies(block["sector"], prefix="sec",
                                     drop_first=True, dtype=float)
            design = pd.concat([design, dummies], axis=1)
        design = sm.add_constant(design, has_constant="add")
        model = sm.OLS(block[target].values, design.values).fit()
        coefficients.append(pd.Series(model.params, index=design.columns))
        r_squared.append(model.rsquared)

    if not coefficients:
        return pd.DataFrame(), np.nan
    panel_coefficients = pd.DataFrame(coefficients)
    summary = []
    for name in panel_coefficients.columns:
        mean, t_stat = newey_west_t(panel_coefficients[name], lags)
        summary.append({"variable": name, "coefficient": mean, "t": t_stat})
    return (pd.DataFrame(summary).set_index("variable"),
            float(np.mean(r_squared)))


HORIZON = "12m"
TARGET = f"forward_return_{HORIZON}"
LAGS = OVERLAP_QUARTERS[HORIZON]

# Drop the near-duplicates: the blended upside is a weighted average of the
# per-model ones, so including both makes the design matrix nearly singular
# and splits one effect across collinear columns.
FM_LOCAL = ["upside_blended_rank", "upside_multiples_rank", "earnings_yield_rank",
            "ebitda_yield_rank", "book_yield_rank", "roic_ttm_rank",
            "ebit_margin_ttm_rank", "confidence_rank"]

local_only, r2_local = fama_macbeth(data, FM_LOCAL, TARGET, LAGS)
with_global, r2_global = fama_macbeth(
    data, FM_LOCAL + ["log_size_rank"], TARGET, LAGS, sector_dummies=True)

print(f"Fama-MacBeth, {HORIZON} forward return, Newey-West lag {LAGS}\n")
comparison = local_only.join(
    with_global, how="outer", lsuffix="_local", rsuffix="_global")
print(comparison.loc[[c for c in comparison.index
                      if not c.startswith("sec_")]]
      .round(4).to_string())
print(f"\nmean cross-sectional R^2 — local only: {r2_local:.1%}, "
      f"with size + sector dummies: {r2_global:.1%}")
''')

code(r'''
# --- sector-neutral target: does the signal survive inside the sector? ---
neutral = data.copy()
neutral["target_neutral"] = (
    neutral[TARGET]
    - neutral.groupby(["quarter", "sector"])[TARGET].transform("mean"))

sector_neutral, r2_neutral = fama_macbeth(neutral, FM_LOCAL, "target_neutral", LAGS)
print("Local features against a sector-neutral target")
print(sector_neutral.round(4).to_string())
print(f"\nmean cross-sectional R^2: {r2_neutral:.1%}")

significant = sector_neutral[sector_neutral["t"].abs() > 2]
print(f"\n{len(significant)} of {len(sector_neutral)-1} local features keep "
      f"|t| > 2 once sector is removed:")
print(", ".join(significant.index.drop("const", errors="ignore")))
''')

md(r'''
## 6. Non-linear cross-check

A gradient-boosted tree, trained on an **expanding window** and tested on the
next unseen quarter, repeatedly. Random cross-validation would be invalid here:
shuffling rows lets the model learn from the future and from the same company's
adjacent filings, and would report a flattering score that no live signal could
reproduce.

Its purpose is diagnostic — if a flexible model finds far more than the linear
one, the relationship is non-linear or interactive; if it finds about the same,
the linear picture is the honest one.

Expect **out-of-sample R² to be negative** while the IC is positive. That is not
a failure, it is the normal shape of return prediction: the model orders stocks
correctly but cannot predict the *level* of a return, and R² punishes it against
a benchmark — the realised mean of the test quarter — that no forecaster could
have known. **IC is the metric that matters here; R² is reported so it is not
quietly omitted.**
''')

code(r'''
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score
from scipy.stats import spearmanr

features = FM_LOCAL + ["log_size_rank"]
model_frame = data.dropna(subset=features + [TARGET]).copy()
model_frame["sector_code"] = model_frame["sector"].astype("category").cat.codes
model_frame["industry_code"] = model_frame["industry"].astype("category").cat.codes
features_full = features + ["sector_code", "industry_code"]

quarters = np.sort(model_frame["quarter"].unique())
# Train on everything up to a quarter, test on the next. The first eight
# quarters are held back as the minimum training window.
folds = []
for split in range(8, len(quarters) - 1):
    train = model_frame[model_frame["quarter"] <= quarters[split]]
    test = model_frame[model_frame["quarter"] == quarters[split + 1]]
    if len(test) < 50:
        continue
    estimator = HistGradientBoostingRegressor(
        max_depth=4, max_iter=250, learning_rate=0.05,
        min_samples_leaf=60, l2_regularization=1.0, random_state=0)
    estimator.fit(train[features_full], train[TARGET])
    prediction = estimator.predict(test[features_full])
    folds.append({
        "quarter": str(quarters[split + 1]),
        "n_train": len(train), "n_test": len(test),
        "r2": r2_score(test[TARGET], prediction),
        "ic": spearmanr(prediction, test[TARGET]).statistic,
    })

fold_results = pd.DataFrame(folds)
mean_ic, t_ic = newey_west_t(fold_results.set_index("quarter")["ic"], LAGS)
print(f"Walk-forward: {len(fold_results)} out-of-sample quarters")
print(f"  mean out-of-sample IC : {mean_ic:+.3f}  (Newey-West t = {t_ic:.2f})")
print(f"  positive IC quarters  : {(fold_results['ic'] > 0).mean():.0%}")
print(f"  mean out-of-sample R^2: {fold_results['r2'].mean():+.3f}")
fold_results.tail(8).round(3).to_string(index=False)
''')

code(r'''
# Permutation importance on the final fold - measured on unseen data, so it
# reports what actually generalised rather than what the model leaned on to fit.
final_train = model_frame[model_frame["quarter"] <= quarters[-2]]
final_test = model_frame[model_frame["quarter"] == quarters[-1]]
estimator = HistGradientBoostingRegressor(
    max_depth=4, max_iter=250, learning_rate=0.05, min_samples_leaf=60,
    l2_regularization=1.0, random_state=0).fit(
        final_train[features_full], final_train[TARGET])

importance = permutation_importance(
    estimator, final_test[features_full], final_test[TARGET],
    n_repeats=8, random_state=0, scoring="r2")
ranking = pd.Series(importance.importances_mean,
                    index=features_full).sort_values()

fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
axes[0].barh(ranking.index, ranking.values, color=ACCENT)
axes[0].axvline(0, color="black", lw=1)
axes[0].set_title("Permutation importance, out-of-sample")

axes[1].bar(fold_results["quarter"], fold_results["ic"],
            color=[UP if v > 0 else DOWN for v in fold_results["ic"]])
axes[1].axhline(mean_ic, color=ACCENT, ls="--", label=f"mean {mean_ic:+.3f}")
axes[1].axhline(0, color="black", lw=1)
axes[1].tick_params(axis="x", rotation=90)
axes[1].set_title("Out-of-sample IC by quarter")
axes[1].legend()
plt.tight_layout()
plt.show()
''')

md(r'''
## 7. What this does and does not establish

**Valuation signals predict; quality and the engine's self-assessment do not.**
Seven features clear |t| > 2 at twelve months, and every one of them is a
cheapness measure: sales yield (IC 0.131), the peer-multiple upside (0.089), the
blended fair value (0.084), book yield (0.082), and the residual-income, FCFF and
FCFE upsides (0.074 / 0.066 / 0.053). Every profitability ratio — ROE, ROIC, EBIT
and net margin — is indistinguishable from noise, and so are `confidence`,
`weighted_reliability`, `models_used` and `model_agreement`. **The engine's
opinion of its own reliability carries no return information.**

**Size is the one global determinant that acts like a signal.** `log_size` has an
IC of −0.079 (t = −2.56): smaller companies outperformed. Sector and industry
behave differently — as *groupings* rather than rankings. Sector alone explains
0.5% of twelve-month return variance and industry 2.5%, but interacted with time,
`quarter × industry` explains **19.8%**. Which industry you were in barely
mattered on average; which industry you were in *in a given quarter* mattered a
great deal.

**The local signal survives the removal of sector.** Against a sector-neutral
target the blended upside keeps a coefficient of +0.073 (t = 2.75) and five other
local features stay significant, so this is not a disguised sector bet.

**Where it works is uneven.** The blended upside has an IC of 0.143 in Utilities,
0.114 in Financial Services and 0.106 in Energy — but **0.018 in Technology
(t = 0.50), where it is worthless.** By size it is concentrated in micro caps
(0.115, t = 4.83), which is also where it is least tradeable.

**A sign flip worth understanding.** Earnings yield is mildly positive on its own
(IC 0.033, not significant) yet takes a *negative* coefficient in the
multivariate regression (−0.168, t = −3.26). That is collinearity, not a finding:
sales yield and book yield carry the cheapness, and earnings yield is left
absorbing what they leave behind. Read the univariate table for what a feature
does, and the multivariate one only for what it adds.

**The non-linear model finds about the same.** A gradient-boosted tree trained
walk-forward reaches an out-of-sample IC of +0.102 (t = 2.63), positive in 94% of
16 quarters — close to the best single linear feature. There is no large
interaction effect being missed.

**Structural limits that no amount of fitting removes:**

- **Survivorship.** The universe is companies listed *today* that pass the
  current screen. Anything delisted before 2026 is absent, which lifts every
  return in the table. Spreads between groups are far more trustworthy than any
  group's level.
- **Overlapping observations.** A 12-month return sampled quarterly shares three
  quarters of its path with its neighbour. Newey–West corrects the standard
  errors; it cannot manufacture the independent observations that never existed.
  The effective sample is closer to the 29 quarters than to the 61,508 rows.
- **One regime.** 2019–2026 is a single mostly-rising market with one sharp
  drawdown. Nothing here tests a prolonged bear market.
- **No costs.** Not spread, not commission, not borrow on any short leg, not
  the capacity limits that bind hardest in exactly the small-cap names where
  these signals tend to be strongest.
''')

code(r'''
summary = pd.DataFrame({
    "mean IC (12m)": ic_table["IC_12m"],
    "t-stat": ic_table["t_12m"],
}).dropna().sort_values("mean IC (12m)", ascending=False)
summary["verdict"] = np.where(
    summary["t-stat"].abs() > 2,
    np.where(summary["mean IC (12m)"] > 0, "predictive", "inverse"),
    "not distinguishable from noise")

print("12-month univariate verdict, Newey-West adjusted\n")
print(summary.round(3).to_string())
print()
print(f"variance explained by sector alone       : {explained['sector']:.1%}")
print(f"variance explained by industry alone     : {explained['industry']:.1%}")
print(f"Fama-MacBeth R^2, local only             : {r2_local:.1%}")
print(f"Fama-MacBeth R^2, local + size + sector  : {r2_global:.1%}")
print(f"Fama-MacBeth R^2, sector-neutral target  : {r2_neutral:.1%}")
print(f"walk-forward out-of-sample IC (GBM)      : {mean_ic:+.3f} (t={t_ic:.2f})")
''')

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python",
                       "name": "python3"},
        "language_info": {"name": "python", "version": "3.13.0"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

target = BASE / "return_drivers.ipynb"
target.write_text(json.dumps(notebook, indent=1))
print(f"wrote {target.name}: {len(cells)} cells")

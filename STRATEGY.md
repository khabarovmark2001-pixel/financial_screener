# What the research says, and what to do with it

Summary of `return_drivers.ipynb`, and the screen it justifies —
`screen_attractive.py` → `data/attractive.parquet`.

Measured on 61,508 point-in-time observations, 2,617 tickers, 2019-09 to
2026-09, cross-sectionally within quarter, Newey–West adjusted for the overlap
that quarterly sampling of a 12-month return creates.

---

## 1. What predicts, and what does not

**Only cheapness predicted anything.** Seven features cleared |t| > 2 against
12-month forward returns, and every one is a valuation measure:

| Feature | IC | t |
|---|---|---|
| Sales yield (1 ÷ P/S) | 0.131 | 3.51 |
| Peer-multiple upside | 0.089 | 3.76 |
| Blended fair value upside | 0.084 | 4.00 |
| Book yield (1 ÷ P/B) | 0.082 | 2.05 |
| Residual-income upside | 0.074 | 5.07 |
| FCFF upside | 0.066 | 3.38 |
| FCFE upside | 0.053 | 2.78 |

**Nothing else did.** ROE, ROIC, EBIT margin and net margin are all
indistinguishable from noise. So are `confidence`, `weighted_reliability`,
`models_used` and `model_agreement` — **the engine's own assessment of how much
to trust itself carries no return information.** It remains a useful diagnostic
for reading a single valuation; it is not a selection criterion.

**Size is the only global variable that behaves like a signal**: IC −0.079
(t = −2.56). Smaller companies outperformed.

**Sector and industry are groupings, not signals.** Sector explains 0.5% of
12-month return variance and industry 2.5%. But `quarter × industry` explains
**19.8%** — *which* industry barely mattered on average; which industry *in a
given quarter* mattered enormously. That is a timing effect this screen makes
no attempt to capture.

---

## 2. Practical rules that follow

### Rule 1 — Rank cheapness inside the industry, never across the market

Drug distributors book the entire drug price as revenue on roughly a 1% margin,
so their price-to-sales is structurally near 0.2 and always will be. Ranked
against the rest of Healthcare, McKesson and Cardinal Health screened in the top
five and the model claimed **+523%** and **+382%** upside. Ranked against other
distributors they are ordinary. Both disappeared from the list the moment
ranking moved to industry level, which is the correct answer.

### Rule 2 — Treat a huge claimed upside as a warning, not an opportunity

Returns rose monotonically through decile 7 and then **fell** in deciles 8 and 9,
where median claimed upside was +147% and +301%. Past roughly +300%, a large
upside stops being evidence that a stock is cheap and starts being evidence that
the model does not describe the business. The screen flags these rather than
ranking them.

### Rule 3 — Do not use the screen in Technology

The signal's IC in Technology is **0.018 with t = 0.50** — indistinguishable
from zero. It is also weak in Real Estate (0.042) and Industrials (0.048). It
works in Utilities (0.143), Financial Services (0.114) and Energy (0.106).
A high score on a technology stock is not evidence of anything.

### Rule 4 — Skip anything with negative book equity

Book yield is meaningless there, residual income is meaningless there, and the
DCF tends to produce a number with no support. 198 names are flagged on this
plus rule 2.

### Rule 5 — Accept that the tradeable version is much weaker

This is the uncomfortable one. The size filter that makes names liquid enough to
own **halves the signal**:

| Filter | IC | Q5−Q1 (12m) |
|---|---|---|
| Composite, all names | +0.076 (t 4.82) | +5.8% |
| + traps excluded | +0.069 (t 4.00) | +5.4% |
| + reliable sectors only | +0.081 (t 4.73) | +6.7% |
| **+ market cap ≥ $2bn** | **+0.049 (t 2.46)** | **+3.1%** |

And on that final filtered set the quintiles are **not monotonic** — +5.7%,
+8.2%, +8.0%, +10.1%, +8.8%. The fifth quintile is *below* the fourth.

### Rule 6 — The composite is not obviously better than one ratio

| | IC | t | Q5−Q1 |
|---|---|---|---|
| 7-feature composite | +0.076 | 4.82 | +5.8% |
| **Sales yield alone** | **+0.082** | 3.88 | **+7.4%** |
| Blended DCF alone | +0.068 | 4.96 | +4.4% |

The composite is *more consistent* across quarters (higher t) but not stronger.
Anyone who wants one number rather than seven can use price-to-sales, ranked
within industry, and give up very little.

---

## 3. What currently screens as attractive

Score is the average industry-relative percentile across the seven features.
Run `python3 screen_attractive.py report` to regenerate after a data update.

### Mid and large cap (≥ $2bn) — 970 candidates, weaker signal but ownable

| Ticker | Company | Sector | Score | Cap | Price | Fair value | P/S | P/B | P/E |
|---|---|---|---|---|---|---|---|---|---|
| BSBR | Banco Santander Brasil | Financials | 1.00 | $48.6bn | $5.89 | $21.12 | 0.6 | 0.4 | 3.8 |
| BCS | Barclays | Financials | 0.98 | $91.1bn | $26.44 | $34.69 | 2.2 | 0.9 | 8.7 |
| BHF | Brighthouse Financial | Financials | 0.96 | $3.6bn | $51.15 | $166.29 | 0.5 | 0.5 | 4.3 |
| FMS | Fresenius Medical Care | Healthcare | 0.93 | $12.8bn | $22.33 | $69.70 | 0.6 | 0.9 | 12.0 |
| HE | Hawaiian Electric | Utilities | 0.92 | $2.1bn | $10.33 | $19.50 | 0.7 | 1.2 | 9.6 |
| KOF | Coca-Cola FEMSA | Cons. Defensive | 0.92 | $21.6bn | $109.16 | $394.00 | 1.3 | 2.6 | 16.1 |
| ICLR | ICON plc | Healthcare | 0.92 | $12.8bn | $159.51 | $283.40 | 1.5 | 1.4 | 38.8 |
| DG | Dollar General | Cons. Defensive | 0.91 | $27.8bn | $124.57 | $290.63 | 0.6 | 3.0 | 16.3 |
| KSS | Kohl's | Cons. Cyclical | 0.91 | $2.2bn | $17.52 | $46.30 | 0.1 | 0.5 | 8.1 |
| CIG | Energy Co of Minas Gerais | Utilities | 0.91 | $7.6bn | $2.17 | $8.00 | 0.9 | 1.3 | 7.8 |
| WAL | Western Alliance Banc. | Financials | 0.89 | $8.8bn | $79.66 | $87.74 | 2.5 | 1.1 | 8.9 |
| UHS | Universal Health Services | Healthcare | 0.89 | $10.2bn | $175.78 | $606.37 | 0.6 | 1.4 | 6.7 |

Heavily weighted to **financials, healthcare services and utilities** — the
three places the signal actually measured well. That concentration is the
method working, not a coincidence, and it is also a concentration risk if you
buy the list as given.

### Small cap ($300m–$2bn) — 280 candidates, strongest signal, hardest to own

NEXA Resources, Weibo, Winnebago, G-III Apparel, Quanex, Mativ, Teladoc, Teekay,
RPC, PRA Group, Rigel, Qfin. Median dollar volume around $10m/day — a $100k
position is a meaningful share of a day's trading in several of them.

---

## 4. How to use it

1. **As a candidate generator, not a buy list.** IC 0.049 on the tradeable set
   means the ranking is right slightly more often than chance. It narrows 2,173
   names to a few dozen worth reading a 10-K about.
2. **Diversify across the list, not into it.** A signal this weak only shows up
   in aggregate. Ten to twenty positions, not two.
3. **Rebalance slowly.** The IC rises with horizon (0.038 at 1 month, 0.093 at
   12) — this is a slow-correcting valuation effect. Quarterly at most;
   trading it monthly pays costs for a signal that has not moved.
4. **Read the flags before acting.** `negative_book_equity`,
   `implausible_upside`, `signal_reliable_here`, `liquid_enough` are all in
   `attractive.parquet`.

---

## 5. What would change the conclusion

- **Survivorship.** The universe is what is listed *today* and passes the
  current screen. Everything delisted before 2026 is absent, which lifts every
  return in every table. **Spreads between quintiles are trustworthy; the levels
  are not.**
- **One regime.** 2019–2026 is a single mostly-rising market. A value tilt's
  behaviour in a prolonged bear market is untested here.
- **No costs.** No spread, no commission, no borrow, no capacity limit — and
  capacity binds hardest in exactly the small caps where the signal is strongest.
- **29 quarters.** The effective sample is closer to 29 independent observations
  than to 61,508 rows, because overlapping 12-month windows share most of their
  path. Newey–West corrects the standard errors; it cannot create data.

This is model output on a self-built dataset, not investment advice.

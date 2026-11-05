#!/bin/bash
# ---------------------------------------------------------------------------
# Bring every number the web page shows up to date, in dependency order.
#
#   universe -> prices + fundamentals -> market data -> metrics + cash flows
#            -> DCF -> analytics -> restart the app
#
# Each stage reads the parquet the stage above it wrote, so the order is not
# negotiable: running the DCF before the metrics rebuild just re-derives the
# same stale valuations from the same stale inputs.
#
# Usage
#   ./update_data.sh                  full daily update, then restart the app
#   ./update_data.sh --quick          skip the universe re-screen (faster)
#   ./update_data.sh --no-restart     leave the running app alone
#   ./update_data.sh --stage metrics  run one stage and stop
#
# Stages: universe prices market metrics dcf analytics
# ---------------------------------------------------------------------------
set -euo pipefail
cd "$(dirname "$0")" || exit 1

PY=.venv/bin/python
[ -x "$PY" ] || PY=python3

QUICK=0
RESTART=1
ONLY=""

while [ $# -gt 0 ]; do
  case "$1" in
    --quick)      QUICK=1 ;;
    --no-restart) RESTART=0 ;;
    --stage)      ONLY="${2:?--stage needs a stage name}"; RESTART=0; shift ;;
    -h|--help)    sed -n '2,18p' "$0"; exit 0 ;;
    *)            echo "unknown option: $1" >&2; exit 2 ;;
  esac
  shift
done

STARTED=$(date +%s)

step() {                      # step <stage-name> <command...>
  local name="$1"; shift
  if [ -n "$ONLY" ] && [ "$ONLY" != "$name" ]; then return 0; fi
  printf '\n==> %-9s %s\n' "$name" "$(date '+%H:%M:%S')"
  "$@"
}

# 1. The screen. Membership drifts with price and market cap, so re-running it
#    is what lets names enter and leave. Skipped by --quick because it is the
#    one stage whose output changes slowly.
if [ "$QUICK" = 0 ]; then
  step universe "$PY" sharadar_loader.py universe
fi

# 2. Prices and fundamentals. `refresh` re-fetches a short overlapping window
#    rather than starting the day after the last stored session - filings are
#    filtered by FILING date, and the vendor restates rows after publishing.
#    It also drops any no-row cache entry first, so a run that started before
#    the vendor published cannot pin the cache to "nothing new".
step prices "$PY" sharadar_loader.py refresh

# 3. FRED rates/indices plus the gold proxy. Independent of the vendor pull,
#    but the metrics stage discounts with these, so it has to land first.
step market "$PY" valuation_metrics.py market

# 4. Per-ticker valuation history and the cash-flow panel.
step metrics "$PY" valuation_metrics.py metrics

# 5. Intrinsic value. Reads cash_flows.parquet + metrics.parquet from step 4.
step dcf "$PY" dcf_forecast.py run

# 6. The screener table itself, and the sector/industry return series. Reads
#    the DCF summary, and stamps the `as_of` date the page prints in its
#    header - which is why a stale analytics.parquet makes the whole page
#    *look* stale even when the prices underneath it are current.
step engine "$PY" valuation_engine.py run

step engine_history "$PY" valuation_history.py run

step analytics "$PY" analytics.py run

# 7. Restart. Streamlit caches every loader with @st.cache_data(ttl=3600) and
#    the server process outlives this script, so fresh parquet on disk is NOT
#    enough - without a restart the page keeps serving the old frames from
#    memory for up to an hour.
if [ "$RESTART" = 1 ]; then
  printf '\n==> restart   %s\n' "$(date '+%H:%M:%S')"
  ./stop_app.sh || true
  ./start_app.sh
fi

printf '\nDone in %ds.\n' "$(( $(date +%s) - STARTED ))"
"$PY" - <<'PYEOF'
import pandas as pd, pathlib
print("\nLatest date now stored:")
for name, column in (("stocks", "date"), ("metrics", "date"),
                     ("dcf_summary", "market_price_date"),
                     ("market_data", "date"), ("analytics", "as_of")):
    path = pathlib.Path("data") / f"{name}.parquet"
    if not path.exists():
        print(f"  {name:<12} missing"); continue
    try:
        value = pd.read_parquet(path, columns=[column])[column].max()
        print(f"  {name:<12} {pd.Timestamp(value).date()}")
    except Exception as exc:
        print(f"  {name:<12} ? ({exc})")
PYEOF

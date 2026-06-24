from pathlib import Path
import pandas as pd

latest_path = Path("data/latest/psx_all_index_constituents.csv")
history_path = Path("data/history/psx_all_index_constituents_history.csv")

if not latest_path.exists():
    raise SystemExit(f"Latest CSV not found: {latest_path}")

latest = pd.read_csv(latest_path, engine="python", on_bad_lines="skip")

if "ScrapeTimestamp" not in latest.columns:
    raise SystemExit("ScrapeTimestamp column missing in latest CSV")

if "SYMBOL" not in latest.columns:
    raise SystemExit("SYMBOL column missing in latest CSV")

latest["SYMBOL"] = latest["SYMBOL"].astype(str).str.upper().str.strip()
latest["_ts"] = pd.to_datetime(latest["ScrapeTimestamp"], errors="coerce")
latest = latest.dropna(subset=["_ts"])
latest["_date"] = latest["_ts"].dt.strftime("%Y-%m-%d")

if history_path.exists():
    history = pd.read_csv(history_path, engine="python", on_bad_lines="skip")
    if "ScrapeTimestamp" in history.columns and "SYMBOL" in history.columns:
        history["SYMBOL"] = history["SYMBOL"].astype(str).str.upper().str.strip()
        history["_ts"] = pd.to_datetime(history["ScrapeTimestamp"], errors="coerce")
        history = history.dropna(subset=["_ts"])
        history["_date"] = history["_ts"].dt.strftime("%Y-%m-%d")
        combined = pd.concat([history, latest], ignore_index=True, sort=False)
    else:
        combined = latest.copy()
else:
    combined = latest.copy()

# Keep only latest snapshot per SYMBOL per trading date
combined = combined.sort_values(["_date", "SYMBOL", "_ts"])
combined = combined.drop_duplicates(subset=["_date", "SYMBOL"], keep="last")

# Keep only last 10 trading dates to keep GitHub file small
last_dates = sorted(combined["_date"].dropna().unique())[-10:]
combined = combined[combined["_date"].isin(last_dates)].copy()

combined = combined.drop(columns=["_ts", "_date"], errors="ignore")

history_path.parent.mkdir(parents=True, exist_ok=True)
combined.to_csv(history_path, index=False)

print(f"✅ Permanent history updated: {history_path}")
print(f"✅ Rows: {len(combined)}")
print(f"✅ Dates: {last_dates}")

from pathlib import Path
import pandas as pd

latest_path = Path("data/latest/psx_all_index_constituents.csv")
history_path = Path("data/history/psx_all_index_constituents_history.csv")

if not latest_path.exists():
    raise SystemExit(f"Latest CSV not found: {latest_path}")

latest = pd.read_csv(latest_path, engine="python", on_bad_lines="skip")

required = ["SYMBOL", "ScrapeTimestamp"]
for col in required:
    if col not in latest.columns:
        raise SystemExit(f"{col} column missing in latest CSV")

latest["SYMBOL"] = latest["SYMBOL"].astype(str).str.upper().str.strip()
latest["ScrapeTimestamp"] = latest["ScrapeTimestamp"].astype(str).str.strip()
latest["_ts"] = pd.to_datetime(latest["ScrapeTimestamp"], errors="coerce")
latest = latest.dropna(subset=["_ts"])

if history_path.exists():
    history = pd.read_csv(history_path, engine="python", on_bad_lines="skip")

    if "SYMBOL" in history.columns and "ScrapeTimestamp" in history.columns:
        history["SYMBOL"] = history["SYMBOL"].astype(str).str.upper().str.strip()
        history["ScrapeTimestamp"] = history["ScrapeTimestamp"].astype(str).str.strip()
        history["_ts"] = pd.to_datetime(history["ScrapeTimestamp"], errors="coerce")
        history = history.dropna(subset=["_ts"])
        combined = pd.concat([history, latest], ignore_index=True, sort=False)
    else:
        combined = latest.copy()
else:
    combined = latest.copy()

# Save ALL unique snapshots.
# One row per SYMBOL per exact ScrapeTimestamp.
combined = combined.sort_values(["_ts", "SYMBOL"])
combined = combined.drop_duplicates(
    subset=["ScrapeTimestamp", "SYMBOL"],
    keep="last"
)

combined = combined.drop(columns=["_ts"], errors="ignore")

history_path.parent.mkdir(parents=True, exist_ok=True)
combined.to_csv(history_path, index=False)

dates = pd.to_datetime(combined["ScrapeTimestamp"], errors="coerce").dt.strftime("%Y-%m-%d")

print(f"✅ Permanent full history updated: {history_path}")
print(f"✅ Total rows: {len(combined)}")
print(f"✅ Unique trading dates: {sorted(dates.dropna().unique())}")
print(f"✅ Latest timestamp: {combined['ScrapeTimestamp'].astype(str).max()}")

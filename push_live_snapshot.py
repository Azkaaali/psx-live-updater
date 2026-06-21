import os
import json
import urllib.request
import importlib.util
from pathlib import Path

import pandas as pd

UPLOAD_URL = "https://azkaalii-psx-ai-backend.hf.space/upload-live-snapshot"
UPLOAD_TOKEN = os.environ.get("UPLOAD_TOKEN", "psx-live-123")

SCRAPER_PATH = Path(__file__).resolve().parent / "psx_full_backup (1)" / "psx_v4.py"

CSV_CANDIDATES = [
    Path.home() / "Desktop/backend/psx_full_backup (1)/output/psx_all_index_constituents.csv",
    Path.home() / "Desktop/backend/psx_full_backup (1)/psx_all_index_constituents.csv",
    Path.home() / "Desktop/backend/output/psx_all_index_constituents.csv",
    Path.home() / "Desktop/psx-ai-backend-hf-clean/output/psx_all_index_constituents.csv",
]


def find_column(df, names):
    lower = {c.lower().strip(): c for c in df.columns}
    for name in names:
        key = name.lower().strip()
        if key in lower:
            return lower[key]
    return None


def run_psx_v4_scraper():
    if not SCRAPER_PATH.exists():
        raise FileNotFoundError(f"psx_v4.py not found: {SCRAPER_PATH}")

    spec = importlib.util.spec_from_file_location("psx_v4_local", SCRAPER_PATH)
    psx_v4 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(psx_v4)

    if not hasattr(psx_v4, "scrape_all_index_constituents"):
        raise AttributeError("scrape_all_index_constituents() not found inside psx_v4.py")

    print("Running your real psx_v4.py scraper...")
    psx_v4.scrape_all_index_constituents()
    print("Scraper completed.")


def get_latest_csv_path():
    import pandas as pd

    roots = [
        Path(__file__).resolve().parent,
        Path.cwd(),
        Path("/home/runner/work/psx-live-updater/psx-live-updater"),
        Path("/home/runner/Desktop/backend"),
        Path("/home/runner/Desktop/backend/output"),
        Path.home() / "Desktop" / "backend",
        Path.home() / "Desktop" / "backend" / "output",
    ]

    candidates = []
    seen = set()

    for root in roots:
        if not root.exists():
            continue

        for csv_path in root.rglob("psx_all_index_constituents.csv"):
            try:
                resolved = csv_path.resolve()

                if resolved in seen:
                    continue

                seen.add(resolved)

                if resolved.is_file() and resolved.stat().st_size > 0:
                    df = pd.read_csv(resolved, nrows=5)

                    timestamp = ""
                    if "ScrapeTimestamp" in df.columns:
                        timestamp = str(df["ScrapeTimestamp"].dropna().astype(str).max())

                    candidates.append({
                        "path": resolved,
                        "mtime": resolved.stat().st_mtime,
                        "size": resolved.stat().st_size,
                        "timestamp": timestamp,
                    })

            except Exception as e:
                print("CSV candidate skipped:", csv_path, e)

    print("CSV candidates found:")
    for item in candidates:
        print(
            " -",
            item["path"],
            "| timestamp=",
            item["timestamp"],
            "| mtime=",
            item["mtime"],
            "| size=",
            item["size"],
        )

    if not candidates:
        raise FileNotFoundError("No psx_all_index_constituents.csv found after scraper run.")

    with_timestamp = [item for item in candidates if item["timestamp"]]

    if with_timestamp:
        latest_item = max(with_timestamp, key=lambda item: item["timestamp"])
    else:
        latest_item = max(candidates, key=lambda item: item["mtime"])

    latest_csv = latest_item["path"]

    print("Using latest CSV:", latest_csv)
    print("Selected CSV timestamp:", latest_item["timestamp"])

    return latest_csv


def get_latest_snapshot_rows(csv_path):
    df = pd.read_csv(csv_path, engine="python", on_bad_lines="skip")

    ts_col = find_column(df, ["ScrapeTimestamp", "timestamp"])
    symbol_col = find_column(df, ["SYMBOL", "Symbol", "symbol", "SCRIP", "Scrip"])

    if ts_col is None:
        raise ValueError(f"Timestamp column not found. Columns: {list(df.columns)}")

    if symbol_col is None:
        raise ValueError(f"Symbol column not found. Columns: {list(df.columns)}")

    df = df.dropna(subset=[ts_col, symbol_col]).copy()
    df[ts_col] = df[ts_col].astype(str)

    latest_ts = df[ts_col].max()
    latest = df[df[ts_col] == latest_ts].copy()

    print(f"CSV used: {csv_path}")
    print(f"Latest timestamp: {latest_ts}")
    print(f"Rows uploading: {len(latest)}")

    return latest.to_dict(orient="records")


def upload_rows(rows):
    payload = json.dumps({"rows": rows}).encode("utf-8")

    req = urllib.request.Request(
        UPLOAD_URL,
        data=payload,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "X-Upload-Token": UPLOAD_TOKEN,
        },
    )

    with urllib.request.urlopen(req, timeout=180) as res:
        print(res.read().decode("utf-8"))


if __name__ == "__main__":
    run_psx_v4_scraper()
    csv_path = get_latest_csv_path()
    rows = get_latest_snapshot_rows(csv_path)
    upload_rows(rows)

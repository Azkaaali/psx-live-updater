from pathlib import Path
import shutil
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent
DEST_DIR = REPO_ROOT / "data" / "latest"
DEST_CSV = DEST_DIR / "psx_all_index_constituents.csv"

SEARCH_ROOTS = [
    REPO_ROOT,
    Path.cwd(),
    Path("/home/runner/Desktop/backend"),
    Path("/home/runner/Desktop/backend/output"),
    Path.home() / "Desktop" / "backend",
    Path.home() / "Desktop" / "backend" / "output",
]

def csv_timestamp(csv_path: Path) -> str:
    try:
        df = pd.read_csv(csv_path, nrows=10)
        for col in df.columns:
            clean = str(col).lower().replace(" ", "").replace("_", "")
            if clean in ["scrapetimestamp", "timestamp", "updatedat", "datetime"]:
                values = df[col].dropna().astype(str)
                if not values.empty:
                    return values.max()
    except Exception as e:
        print("Timestamp read failed:", csv_path, e)

    return ""

def find_latest_csv() -> Path:
    candidates = []
    seen = set()

    for root in SEARCH_ROOTS:
        if not root.exists():
            continue

        for csv_path in root.rglob("psx_all_index_constituents.csv"):
            try:
                resolved = csv_path.resolve()

                if resolved in seen:
                    continue

                seen.add(resolved)

                if resolved.is_file() and resolved.stat().st_size > 0:
                    candidates.append({
                        "path": resolved,
                        "timestamp": csv_timestamp(resolved),
                        "mtime": resolved.stat().st_mtime,
                        "size": resolved.stat().st_size,
                    })

            except Exception as e:
                print("Skipped candidate:", csv_path, e)

    print("CSV candidates:")
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
        raise FileNotFoundError("No psx_all_index_constituents.csv found.")

    with_ts = [item for item in candidates if item["timestamp"]]

    if with_ts:
        selected = max(with_ts, key=lambda item: (item["timestamp"], item["mtime"]))
    else:
        selected = max(candidates, key=lambda item: item["mtime"])

    print("Selected latest CSV:", selected["path"])
    print("Selected timestamp:", selected["timestamp"])

    return selected["path"]

def main():
    latest_csv = find_latest_csv()

    DEST_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(latest_csv, DEST_CSV)

    print("Persistent CSV saved to:", DEST_CSV)

if __name__ == "__main__":
    main()

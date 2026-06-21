# =============================================================================
# PSX COMPLETE PIPELINE - MERGED (Scraper + Preprocessing + Training)
# Binary Classification: BUY vs SELL only (HOLD removed for better accuracy)
# =============================================================================
# USAGE:
#   python psx_pipeline_merged.py --mode scrape        # Only scrape data
#   python psx_pipeline_merged.py --mode train         # Preprocess + Train
#   python psx_pipeline_merged.py --mode all           # Run everything together
# =============================================================================

import os
import re
import csv
import sys
import json
import argparse
import warnings
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from typing import List, Tuple, Optional

warnings.filterwarnings("ignore")

# =============================================================================
# GLOBAL CONFIG
# =============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

OUTPUT_DIR        = os.path.join(BASE_DIR, "output")
DEBUG_DIR         = os.path.join(OUTPUT_DIR, "debug_dps")
PREPROCESSED_DIR  = os.path.join(OUTPUT_DIR, "preprocessed_v3")
MODEL_DIR         = os.path.join(OUTPUT_DIR, "model_output_xgb_v3")

for d in [OUTPUT_DIR, DEBUG_DIR, PREPROCESSED_DIR, MODEL_DIR]:
    os.makedirs(d, exist_ok=True)

# --- Scraper paths ---
INDICES_URL   = "https://dps.psx.com.pk/indices"
LISTINGS_URL  = "https://dps.psx.com.pk/listings"
ALL_CONS_CSV  = os.path.join(OUTPUT_DIR, "psx_all_index_constituents.csv")
LISTINGS_CSV  = os.path.join(OUTPUT_DIR, "psx_listings.csv")
HEADLESS      = True
NAV_TIMEOUT_MS   = 150_000
TABLE_WAIT_MS    = 60_000

# --- Preprocessing config ---
TARGET_HORIZON      = 1        # 1 row ahead — works even with few scrape runs
UP_THRESHOLD        = 0.002    # +0.2% = BUY (loosened for sparse data)
DOWN_THRESHOLD      = -0.002   # -0.2% = SELL (loosened for sparse data)
EXPECTED_INTERVAL_MIN = 5
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15

KNOWN_INDICES = [
    "KSE100","KSE30","MIII","ALLSHR","KMI30","KMIALLSHR",
    "OCTB","NBPPGI","BKTI","UPP9","NITPGI","JSGBKTI","ACI"
]

# --- Binary label map (HOLD removed) ---
LABEL_MAP = {0: "SELL", 1: "BUY"}

# =============================================================================
# SECTION 1: SCRAPER HELPERS
# =============================================================================

def now_ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def safe_ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def save_debug(page, tag: str) -> None:
    ts = safe_ts()
    try:
        page.screenshot(path=os.path.join(DEBUG_DIR, f"{ts}_{tag}.png"), full_page=True)
    except Exception:
        pass
    try:
        html = page.content()
        with open(os.path.join(DEBUG_DIR, f"{ts}_{tag}.html"), "w", encoding="utf-8") as f:
            f.write(html)
    except Exception:
        pass

def load_seen_rows(csv_path: str) -> set:
    seen = set()
    if not (os.path.exists(csv_path) and os.path.getsize(csv_path) > 0):
        return seen
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        try:
            next(reader)
        except StopIteration:
            return seen
        for row in reader:
            if row:
                seen.add(tuple(row))
    return seen

def append_rows(csv_path: str, headers: List[str], rows: List[List[str]]) -> int:
    if not rows:
        return 0
    file_exists = os.path.exists(csv_path) and os.path.getsize(csv_path) > 0
    with open(csv_path, "a", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        if not file_exists:
            w.writerow(headers)
        w.writerows(rows)
    return len(rows)

def _close_overlays_best_effort(page) -> None:
    candidates = [
        "button:has-text('Accept')", "button:has-text('I Agree')",
        "button:has-text('Agree')", "button:has-text('Close')",
        "[aria-label='Close']", ".close", ".btn-close",
    ]
    for sel in candidates:
        try:
            loc = page.locator(sel).first
            if loc.count() > 0 and loc.is_visible():
                loc.click(timeout=1500)
                page.wait_for_timeout(400)
                break
        except Exception:
            pass

def _new_context(p):
    browser = p.chromium.launch(
        headless=HEADLESS,
        args=["--disable-blink-features=AutomationControlled","--no-sandbox","--disable-dev-shm-usage"],
    )
    context = browser.new_context(
        user_agent=(
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36"
        ),
        viewport={"width": 1366, "height": 768},
        locale="en-US",
    )
    context.add_init_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined});")
    return browser, context

def _goto_safe(page, url: str) -> None:
    try:
        page.goto(url, wait_until="domcontentloaded", timeout=NAV_TIMEOUT_MS)
    except Exception:
        page.goto(url, wait_until="load", timeout=NAV_TIMEOUT_MS)
    page.wait_for_timeout(1200)
    _close_overlays_best_effort(page)

def _extract_table_headers_and_rows(table_locator) -> Tuple[List[str], List[List[str]]]:
    header_cells = table_locator.locator("thead tr").first.locator("th, td")
    if header_cells.count() == 0:
        header_cells = table_locator.locator("tr").first.locator("th, td")
    headers = [header_cells.nth(i).inner_text().strip() for i in range(header_cells.count())]
    rows_locator = table_locator.locator("tbody tr")
    rows: List[List[str]] = []
    for r_i in range(rows_locator.count()):
        row_cells = rows_locator.nth(r_i).locator("td")
        if row_cells.count() == 0:
            continue
        rows.append([row_cells.nth(c).inner_text().strip() for c in range(row_cells.count())])
    return headers, rows

def _find_best_table_locator_by_headers(page, required_headers: List[str]):
    page.wait_for_selector("table", timeout=TABLE_WAIT_MS)
    tables = page.locator("table")
    req = [h.lower() for h in required_headers]
    best_score, best_table = -1, None
    for i in range(tables.count()):
        t = tables.nth(i)
        headers, _ = _extract_table_headers_and_rows(t)
        score = sum(1 for r in req if r in [h.lower() for h in headers])
        if score > best_score:
            best_score, best_table = score, t
    if best_table is None:
        raise RuntimeError(f"No table found for headers {required_headers}")
    return best_table

def _is_next_disabled(page) -> bool:
    try:
        li = page.locator("li.paginate_button.next").first
        if li.count() > 0:
            cls = (li.get_attribute("class") or "").lower()
            return "disabled" in cls
    except Exception:
        pass
    return False

def _click_next_if_possible(page) -> bool:
    if _is_next_disabled(page):
        return False
    for sel in ["li.paginate_button.next a", "a:has-text('Next')", "button:has-text('Next')"]:
        try:
            btn = page.locator(sel).first
            if btn.count() > 0 and btn.is_visible():
                btn.click(timeout=5000)
                page.wait_for_timeout(900)
                return True
        except Exception:
            continue
    return False

def _page_signature(rows: List[List[str]]) -> str:
    if not rows:
        return "0|EMPTY|EMPTY"
    return f"{len(rows)}|{rows[0][0] if rows[0] else ''}|{rows[-1][0] if rows[-1] else ''}"

def _wait_table_stable(table_locator, tries: int = 6, delay_ms: int = 500) -> None:
    last, same = -1, 0
    page = table_locator.page
    for _ in range(tries):
        try:
            cnt = table_locator.locator("tbody tr").count()
            if cnt == last:
                same += 1
                if same >= 2:
                    return
            else:
                same = 0
            last = cnt
            page.wait_for_timeout(delay_ms)
        except Exception:
            break

# =============================================================================
# SECTION 2: SCRAPER FUNCTIONS
# =============================================================================

def scrape_all_index_constituents() -> None:
    """Scrape constituents for every PSX index (paginated)."""
    from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError

    seen = load_seen_rows(ALL_CONS_CSV)

    with sync_playwright() as p:
        browser, context = _new_context(p)
        page = context.new_page()
        try:
            _goto_safe(page, INDICES_URL)
            page.wait_for_function(
                "() => document.querySelectorAll('table tbody tr').length >= 10",
                timeout=TABLE_WAIT_MS,
            )

            indices_table = _find_best_table_locator_by_headers(
                page, required_headers=["Index", "High", "Low", "Current", "Change"]
            )
            summary_rows = indices_table.locator("tbody tr")
            index_codes: List[str] = []
            for i in range(summary_rows.count()):
                link = summary_rows.nth(i).locator("td").first.locator("a.link").first
                if link.count() == 0:
                    continue
                code = (link.get_attribute("data-code") or "").strip()
                if code:
                    index_codes.append(code)

            index_codes = list(dict.fromkeys(index_codes))
            print(f"🧾 Found {len(index_codes)} indices: {index_codes}\n")

            timestamp = now_ts()
            out_headers: Optional[List[str]] = None
            all_new: List[List[str]] = []

            for idx_num, index_code in enumerate(index_codes, start=1):
                print(f"➡️  ({idx_num}/{len(index_codes)}) Constituents for: {index_code}")
                try:
                    # Click the index link
                    link = page.locator(f"a.link[data-code='{index_code}']").first
                    link.scroll_into_view_if_needed(timeout=5000)
                    page.wait_for_timeout(150)
                    link.click(timeout=20_000, force=True)

                    # Wait longer and check URL changed OR modal/panel appeared
                    page.wait_for_timeout(2500)
                    _close_overlays_best_effort(page)

                    # Wait specifically for SYMBOL column to appear in any table
                    try:
                        page.wait_for_function(
                            """() => {
                                const headers = [...document.querySelectorAll('table th, table td')];
                                return headers.some(h => h.innerText.trim().toUpperCase() === 'SYMBOL');
                            }""",
                            timeout=15000
                        )
                    except Exception:
                        print(f"   ⚠️ SYMBOL column not found — skipping {index_code}")
                        _goto_safe(page, INDICES_URL)
                        continue

                    # Extra wait for table to fully load
                    page.wait_for_timeout(800)

                    cons_table = _find_best_table_locator_by_headers(page, required_headers=["SYMBOL", "NAME"])
                    cons_headers, test_rows = _extract_table_headers_and_rows(cons_table)

                    # Verify it is actually a stock table (not index summary)
                    if "SYMBOL" not in [h.upper() for h in cons_headers]:
                        print(f"   ⚠️ Wrong table mili (headers: {cons_headers}) — skipping {index_code}")
                        _goto_safe(page, INDICES_URL)
                        continue

                    print(f"   ✅ Headers: {cons_headers}")

                    # Lock headers on first index — all subsequent must match
                    if out_headers is None:
                        out_headers = ["ScrapeTimestamp", "IndexCode"] + cons_headers
                        print(f"   🔒 Locked headers: {out_headers}")
                    elif cons_headers != out_headers[2:]:
                        # Different column structure — pad/trim to match locked headers
                        print(f"   ⚠️ Header mismatch for {index_code} — will pad/trim rows to match")

                    seen_sigs = set()
                    page_no = 1
                    while True:
                        _wait_table_stable(cons_table)
                        _, rows = _extract_table_headers_and_rows(cons_table)
                        sig = _page_signature(rows)
                        print(f"   📄 Page {page_no}: rows={len(rows)}")

                        if sig in seen_sigs:
                            print(f"   🛑 Stop: page repeated")
                            break
                        seen_sigs.add(sig)

                        for r in rows:
                            row_out = [timestamp, index_code] + r
                            if len(row_out) < len(out_headers):
                                row_out += [""] * (len(out_headers) - len(row_out))
                            if len(row_out) > len(out_headers):
                                row_out = row_out[:len(out_headers)]
                            t = tuple(row_out)
                            if t not in seen:
                                seen.add(t)
                                all_new.append(row_out)

                        if not _click_next_if_possible(page):
                            print("   🛑 Stop: Next disabled\n")
                            break
                        cons_table = _find_best_table_locator_by_headers(page, required_headers=["SYMBOL", "NAME"])
                        page_no += 1

                    _goto_safe(page, INDICES_URL)

                except Exception as e:
                    save_debug(page, f"constituents_fail_{index_code}")
                    print(f"⚠️ Skipped {index_code}: {e}\n")
                    try:
                        _goto_safe(page, INDICES_URL)
                    except Exception:
                        pass
                    continue

            if out_headers is None:
                print("❌ Constituents headers not found.")
                return

            added = append_rows(ALL_CONS_CSV, out_headers, all_new)
            if added:
                print(f"✅ Constituents: {added} new rows saved at {timestamp}")
            else:
                print(f"ℹ️  Constituents: no new rows at {timestamp}")

        except Exception as e:
            save_debug(page, "all_index_constituents_exception")
            print(f"❌ Error: {e}")
        finally:
            try:
                context.close()
            except Exception:
                pass
            browser.close()


def scrape_listings() -> None:
    """Scrape PSX listings from all boards."""
    from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError

    seen = load_seen_rows(LISTINGS_CSV)

    with sync_playwright() as p:
        browser, context = _new_context(p)
        page = context.new_page()
        try:
            _goto_safe(page, LISTINGS_URL)
            timestamp = now_ts()
            tab_names = ["Main Board","GEM Board","Normal Counter","Non-Compliant Segment","Winding-Up Segment"]
            all_new_rows: List[List[str]] = []
            out_headers: Optional[List[str]] = None

            for tab in tab_names:
                try:
                    page.get_by_text(tab, exact=False).click(timeout=8_000)
                    page.wait_for_timeout(700)
                except Exception:
                    continue

                try:
                    page.evaluate("window.scrollTo(0, document.body.scrollHeight);")
                    page.wait_for_timeout(500)
                    page.wait_for_selector("table", timeout=20_000)
                    header_cells = page.locator("table thead tr").first.locator("th, td")
                    table_headers = [header_cells.nth(i).inner_text().strip() for i in range(header_cells.count())]
                    rows_locator = page.locator("table tbody tr")
                    table_rows = []
                    for i in range(rows_locator.count()):
                        row_cells = rows_locator.nth(i).locator("td")
                        if row_cells.count() == 0:
                            continue
                        table_rows.append([row_cells.nth(j).inner_text().strip() for j in range(row_cells.count())])
                except PlaywrightTimeoutError:
                    continue

                if out_headers is None:
                    out_headers = ["ScrapeTimestamp", "Board"] + table_headers

                for r in table_rows:
                    row = [timestamp, tab] + r
                    if len(row) < len(out_headers):
                        row += [""] * (len(out_headers) - len(row))
                    if len(row) > len(out_headers):
                        row = row[:len(out_headers)]
                    t = tuple(row)
                    if t not in seen:
                        seen.add(t)
                        all_new_rows.append(row)

            if out_headers is None:
                print("❌ Listings: no table found.")
                return

            added = append_rows(LISTINGS_CSV, out_headers, all_new_rows)
            if added:
                print(f"✅ Listings: {added} new rows saved at {timestamp}")
            else:
                print(f"ℹ️  Listings: no new rows at {timestamp}")

        except Exception as e:
            save_debug(page, "listings_exception")
            print(f"❌ Listings error: {e}")
        finally:
            try:
                context.close()
            except Exception:
                pass
            browser.close()


# =============================================================================
# SECTION 3: PREPROCESSING HELPERS
# =============================================================================

def clean_numeric(series):
    return pd.to_numeric(
        series.astype(str)
              .str.replace(",", "", regex=False)
              .str.replace("%", "", regex=False)
              .str.replace("—", "", regex=False)
              .str.replace("-", "", regex=False)
              .str.strip(),
        errors="coerce"
    )

def standardize_symbol(x):
    x = str(x).strip().upper()
    return "".join(ch for ch in x if ch.isalnum() or ch in [".", "-"])

def downcast_numeric(df):
    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = df[col].astype("float32")
    for col in df.select_dtypes(include=["int64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="integer")
    return df

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period, min_periods=period).mean()
    avg_loss = loss.rolling(period, min_periods=period).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    return 100 - (100 / (1 + rs))

def compute_macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line, macd - signal_line

def compute_bollinger(series, window=20, num_std=2):
    ma  = series.rolling(window, min_periods=window).mean()
    std = series.rolling(window, min_periods=window).std()
    upper = ma + num_std * std
    lower = ma - num_std * std
    width  = (upper - lower) / (ma + 1e-9)
    zscore = (series - ma) / (std + 1e-9)
    return ma, upper, lower, width, zscore

def add_time_features(df):
    df["hour"]        = df["timestamp"].dt.hour
    df["minute"]      = df["timestamp"].dt.minute
    df["dayofweek"]   = df["timestamp"].dt.dayofweek
    df["day"]         = df["timestamp"].dt.day
    df["month"]       = df["timestamp"].dt.month
    df["weekofyear"]  = df["timestamp"].dt.isocalendar().week.astype(int)
    df["is_month_start"] = df["timestamp"].dt.is_month_start.astype(int)
    df["is_month_end"]   = df["timestamp"].dt.is_month_end.astype(int)

    conditions = [
        (df["hour"] < 10),
        (df["hour"] >= 10) & (df["hour"] < 12),
        (df["hour"] >= 12) & (df["hour"] < 14),
        (df["hour"] >= 14)
    ]
    df["session_bucket"] = np.select(conditions, [0, 1, 2, 3], default=3)
    df["hour_sin"]   = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"]   = np.cos(2 * np.pi * df["hour"] / 24)
    df["minute_sin"] = np.sin(2 * np.pi * df["minute"] / 60)
    df["minute_cos"] = np.cos(2 * np.pi * df["minute"] / 60)
    df["dow_sin"]    = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["dow_cos"]    = np.cos(2 * np.pi * df["dayofweek"] / 7)
    return df

def winsorize_by_symbol(df, cols, low_q=0.01, high_q=0.99):
    for col in cols:
        if col not in df.columns:
            continue
        low  = df.groupby("symbol")[col].transform(lambda x: x.quantile(low_q))
        high = df.groupby("symbol")[col].transform(lambda x: x.quantile(high_q))
        df[col] = df[col].clip(lower=low, upper=high)
    return df

def make_binary_target(r):
    """
    HOLD removed — only BUY and SELL.
    Return: 1 = BUY, 0 = SELL, NaN = neutral zone (will be dropped)
    """
    if pd.isna(r):
        return np.nan
    if r >= UP_THRESHOLD:
        return 1   # BUY
    elif r <= DOWN_THRESHOLD:
        return 0   # SELL
    return np.nan  # Neutral zone — excluded from training


# =============================================================================
# SECTION 4: PREPROCESSING PIPELINE
# =============================================================================

def run_preprocessing(input_csv: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Load CSV, engineer features, split into BUY/SELL labels.
    Returns: train_df, val_df, test_df, feature_cols
    """
    from sklearn.preprocessing import StandardScaler

    print("\n" + "="*60)
    print("STEP 2: PREPROCESSING")
    print("="*60)

    print("Loading CSV:", input_csv)

    # Robust CSV load — skips bad lines (mixed column counts from scraper)
    df = pd.read_csv(input_csv, engine="python", on_bad_lines="skip")
    print("Raw shape:", df.shape)

    # Drop repeated header rows if any
    first_col = df.columns[0]
    df = df[df[first_col] != first_col].copy()
    df = df.reset_index(drop=True)
    print("After cleaning bad rows:", df.shape)

    # --- Flexible Rename (PSX column names vary) ---
    print("Actual CSV columns:", df.columns.tolist())

    col_map_candidates = {
        "timestamp":    ["ScrapeTimestamp", "Timestamp", "timestamp"],
        "index_code":   ["IndexCode", "Index Code", "index_code", "Index"],
        "symbol":       ["SYMBOL", "Symbol", "symbol", "SCRIP", "Scrip"],
        "name":         ["NAME", "Name", "name", "COMPANY", "Company Name"],
        "ldcp":         ["LDCP", "Ldcp", "ldcp", "PREV CLOSE", "Prev Close"],
        "current":      ["CURRENT", "Current", "current", "LAST", "Last Price", "PRICE"],
        "change":       ["CHANGE", "Change", "change"],
        "change_pct":   ["CHANGE (%)", "% Change", "Change (%)", "change_pct", "CHG%", "% CHG"],
        "idx_wtg_pct":  ["IDX WTG (%)", "Idx Wtg (%)", "idx_wtg_pct", "INDEX WEIGHT"],
        "idx_point":    ["IDX POINT", "Idx Point", "idx_point", "INDEX POINTS"],
        "volume":       ["VOLUME", "Volume", "volume", "VOL"],
        "freefloat_m":  ["FREEFLOAT (M)", "Free Float (M)", "freefloat_m", "FREE FLOAT"],
        "market_cap_m": ["MARKET CAP (M)", "Market Cap (M)", "market_cap_m", "MKT CAP"],
    }

    rename_map = {}
    for target_col, candidates in col_map_candidates.items():
        for candidate in candidates:
            if candidate in df.columns:
                rename_map[candidate] = target_col
                break

    print("Rename map applied:", rename_map)
    df = df.rename(columns=rename_map)

    # ── Critical check: stop if required columns are missing ──
    missing = [c for c in ["symbol", "current", "timestamp"] if c not in df.columns]
    if missing:
        print(f"\n❌ Required columns missing: {missing}")
        print(f"   Available columns: {df.columns.tolist()}")
        print("   Scraper did not return stock-level data — index summary was scraped instead.")
        print("   Please re-run scraper — it has been fixed.")
        raise ValueError(f"Missing required columns: {missing}")

    # --- Basic cleaning ---
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["symbol"]    = df["symbol"].astype(str).map(standardize_symbol)

    for col in ["ldcp","current","change","change_pct","idx_wtg_pct","idx_point","volume","freefloat_m","market_cap_m"]:
        if col in df.columns:
            df[col] = clean_numeric(df[col])

    df = df.dropna(subset=["timestamp", "symbol", "current"]).copy()
    df = df.drop_duplicates().copy()

    # --- Index membership ---
    df["index_code"] = df["index_code"].astype(str).str.upper().str.strip()
    index_dummies = pd.crosstab(
        [df["symbol"], df["timestamp"]],
        df["index_code"]
    ).reset_index()

    dummy_cols = []
    for c in index_dummies.columns:
        if c not in ["symbol", "timestamp"]:
            new_c = f"in_{str(c).lower()}"
            dummy_cols.append(new_c)
    index_dummies.columns = ["symbol", "timestamp"] + dummy_cols

    membership_cols = [c for c in index_dummies.columns if c.startswith("in_")]
    index_dummies["index_membership_count"] = index_dummies[membership_cols].sum(axis=1)

    for idx in KNOWN_INDICES:
        col = f"in_{idx.lower()}"
        if col not in index_dummies.columns:
            index_dummies[col] = 0

    # --- Collapse to unique symbol+timestamp ---
    agg_dict = {k: v for k, v in {
        "name": "first", "ldcp": "median", "current": "median",
        "change": "median", "change_pct": "median", "idx_wtg_pct": "median",
        "idx_point": "median", "volume": "max", "freefloat_m": "median",
        "market_cap_m": "median"
    }.items() if k in df.columns}

    df = df.groupby(["symbol", "timestamp"], as_index=False).agg(agg_dict)
    df = df.merge(index_dummies, on=["symbol", "timestamp"], how="left")
    df = df.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    print("After dedup:", df.shape)

    # --- Interval consistency ---
    df["time_diff_min"]   = df.groupby("symbol")["timestamp"].diff().dt.total_seconds() / 60.0
    df["interval_ok"]     = df["time_diff_min"].between(EXPECTED_INTERVAL_MIN - 1.5, EXPECTED_INTERVAL_MIN + 1.5)
    df["interval_gap_flag"] = ((~df["interval_ok"]) & df["time_diff_min"].notna()).astype(int)

    # --- Missing values ---
    for col in ["name", "ldcp", "freefloat_m", "market_cap_m"]:
        if col in df.columns:
            df[col] = df.groupby("symbol")[col].ffill().bfill()
    if "volume" in df.columns:
        df["volume"] = df["volume"].fillna(0)

    df = df[df["current"] > 0].copy()
    if "ldcp" in df.columns:
        df.loc[df["ldcp"] <= 0, "ldcp"] = np.nan
        df["ldcp"] = df.groupby("symbol")["ldcp"].ffill().bfill()

    # --- Bad price/volume flags ---
    df["bad_price_flag"] = 0
    if "ldcp" in df.columns:
        ratio = df["current"] / (df["ldcp"] + 1e-9)
        df.loc[(ratio < 0.2) | (ratio > 5.0), "bad_price_flag"] = 1
    df["bad_volume_flag"] = 0
    if "volume" in df.columns:
        df.loc[df["volume"] < 0, "bad_volume_flag"] = 1
    df = df[(df["bad_price_flag"] == 0) & (df["bad_volume_flag"] == 0)].copy()

    # --- Stale detection ---
    df["same_price_prev"] = (df.groupby("symbol")["current"].diff().fillna(0) == 0).astype(int)
    df["same_vol_prev"]   = (df.groupby("symbol")["volume"].diff().fillna(0) == 0).astype(int) if "volume" in df.columns else 0
    df["stale_score"] = (
        df.groupby("symbol")["same_price_prev"].rolling(6, min_periods=1).sum().reset_index(level=0, drop=True)
        + df.groupby("symbol")["same_vol_prev"].rolling(6, min_periods=1).sum().reset_index(level=0, drop=True)
    )
    df["stale_flag"] = (df["stale_score"] >= 10).astype(int)

    # --- Returns + corp action ---
    df["ret_1"]           = df.groupby("symbol")["current"].pct_change()
    df["abs_ret_1"]       = df["ret_1"].abs()
    df["corp_action_flag"] = ((df["abs_ret_1"] > 0.30) & (df["interval_gap_flag"] == 0)).astype(int)

    # --- Winsorize ---
    df = winsorize_by_symbol(df, cols=["current","change","change_pct","volume","idx_point"])

    # --- Feature engineering ---
    g = df.groupby("symbol")

    for lag in [1, 2, 3, 6, 12]:
        df[f"current_lag_{lag}"] = g["current"].shift(lag)
        if "volume" in df.columns:
            df[f"volume_lag_{lag}"] = g["volume"].shift(lag)

    for lag in [1, 2, 3, 6, 12]:
        df[f"return_{lag}"]     = g["current"].pct_change(lag)
        df[f"log_return_{lag}"] = np.log(df["current"] / (g["current"].shift(lag) + 1e-9))

    for w in [3, 6, 12, 24]:
        df[f"roll_mean_{w}"] = g["current"].transform(lambda x: x.rolling(w, min_periods=1).mean())
        df[f"roll_std_{w}"]  = g["current"].transform(lambda x: x.rolling(w, min_periods=2).std())
        df[f"roll_min_{w}"]  = g["current"].transform(lambda x: x.rolling(w, min_periods=1).min())
        df[f"roll_max_{w}"]  = g["current"].transform(lambda x: x.rolling(w, min_periods=1).max())
        if "volume" in df.columns:
            df[f"vol_roll_mean_{w}"] = g["volume"].transform(lambda x: x.rolling(w, min_periods=1).mean())
            df[f"vol_roll_std_{w}"]  = g["volume"].transform(lambda x: x.rolling(w, min_periods=2).std())

    for w in [5, 10, 20]:
        df[f"sma_{w}"]          = g["current"].transform(lambda x: x.rolling(w, min_periods=1).mean())
        df[f"ema_{w}"]          = g["current"].transform(lambda x: x.ewm(span=w, adjust=False).mean())
        df[f"price_to_sma_{w}"] = df["current"] / (df[f"sma_{w}"] + 1e-9)
        df[f"price_to_ema_{w}"] = df["current"] / (df[f"ema_{w}"] + 1e-9)

    for w in [6, 12, 24]:
        df[f"volatility_{w}"]    = g["ret_1"].transform(lambda x: x.rolling(w, min_periods=2).std())
        df[f"realized_vol_{w}"]  = df[f"volatility_{w}"] * np.sqrt(w)

    for w in [3, 6, 12]:
        df[f"momentum_{w}"] = df["current"] - g["current"].shift(w)
        df[f"roc_{w}"]      = (df["current"] / (g["current"].shift(w) + 1e-9)) - 1

    df["rsi_14"]      = g["current"].transform(lambda x: compute_rsi(x, 14))
    df["macd"]        = g["current"].transform(lambda x: compute_macd(x)[0])
    df["macd_signal"] = g["current"].transform(lambda x: compute_macd(x)[1])
    df["macd_hist"]   = g["current"].transform(lambda x: compute_macd(x)[2])

    df["bb_mid"]    = g["current"].transform(lambda x: compute_bollinger(x)[0])
    df["bb_upper"]  = g["current"].transform(lambda x: compute_bollinger(x)[1])
    df["bb_lower"]  = g["current"].transform(lambda x: compute_bollinger(x)[2])
    df["bb_width"]  = g["current"].transform(lambda x: compute_bollinger(x)[3])
    df["bb_zscore"] = g["current"].transform(lambda x: compute_bollinger(x)[4])

    if "volume" in df.columns:
        for w in [6, 12, 24]:
            df[f"volume_ratio_{w}"]  = df["volume"] / (df[f"vol_roll_mean_{w}"] + 1e-9)
            df[f"volume_zscore_{w}"] = (df["volume"] - df[f"vol_roll_mean_{w}"]) / (df[f"vol_roll_std_{w}"] + 1e-9)
        df["volume_spike_flag"] = (df["volume_ratio_12"] > 2.0).astype(int)
    else:
        df["volume_spike_flag"] = 0

    market_ret = df.groupby("timestamp")["ret_1"].mean().rename("market_ret_1")
    df = df.merge(market_ret, on="timestamp", how="left")

    if "volume" in df.columns:
        market_total_vol = df.groupby("timestamp")["volume"].sum().rename("market_total_volume")
        df = df.merge(market_total_vol, on="timestamp", how="left")
        df["volume_market_share"] = df["volume"] / (df["market_total_volume"] + 1e-9)
    else:
        df["volume_market_share"] = 0

    df["excess_return_1"] = df["ret_1"] - df["market_ret_1"]

    df["trade_date"] = df["timestamp"].dt.date
    day_high = df.groupby(["symbol", "trade_date"])["current"].transform("max")
    day_low  = df.groupby(["symbol", "trade_date"])["current"].transform("min")
    df["day_high_dist"]       = (day_high - df["current"]) / (day_high + 1e-9)
    df["day_low_dist"]        = (df["current"] - day_low) / (day_low + 1e-9)
    df["intraday_range_ratio"] = (day_high - day_low) / (day_low + 1e-9)

    df = add_time_features(df)

    # --- Rule-based alert scores ---
    df["alert_buy_score_rule"] = (
        (df["rsi_14"] < 35).astype(int)
        + (df["macd_hist"] > 0).astype(int)
        + (df["excess_return_1"] > 0).astype(int)
        + (df["volume_spike_flag"] == 1).astype(int)
    )
    df["alert_sell_score_rule"] = (
        (df["rsi_14"] > 70).astype(int)
        + (df["macd_hist"] < 0).astype(int)
        + (df["excess_return_1"] < 0).astype(int)
    )
    df["rule_alert_buy"]  = (df["alert_buy_score_rule"] >= 3).astype(int)
    df["rule_alert_sell"] = (df["alert_sell_score_rule"] >= 3).astype(int)

    # --- TARGET (Binary: BUY/SELL, HOLD remove) ---
    df["future_price"]  = g["current"].shift(-TARGET_HORIZON)
    df["future_return"] = (df["future_price"] / (df["current"] + 1e-9)) - 1
    df["target_class"]  = df["future_return"].apply(make_binary_target)

    # --- Final cleanup ---
    df = df.replace([np.inf, -np.inf], np.nan)
    # Drop NaN targets (includes neutral zone)
    df = df.dropna(subset=["future_return", "target_class"]).copy()
    df["target_class"] = df["target_class"].astype(int)

    print(f"Class distribution (0=SELL, 1=BUY):\n{df['target_class'].value_counts()}")

    df["training_usable"] = ((df["stale_flag"] == 0) & (df["corp_action_flag"] == 0)).astype(int)
    trainable_df = df[df["training_usable"] == 1].copy()

    # Fill NaNs in numeric cols
    numeric_cols_all = trainable_df.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols_all:
        trainable_df[col] = trainable_df.groupby("symbol")[col].ffill().bfill()
        trainable_df[col] = trainable_df[col].fillna(trainable_df[col].median())

    trainable_df = downcast_numeric(trainable_df)

    # --- Time split ---
    unique_times = np.array(sorted(trainable_df["timestamp"].unique()))
    n = len(unique_times)
    train_end = int(n * TRAIN_RATIO)
    val_end   = int(n * (TRAIN_RATIO + VAL_RATIO))

    train_df = trainable_df[trainable_df["timestamp"].isin(unique_times[:train_end])].copy()
    val_df   = trainable_df[trainable_df["timestamp"].isin(unique_times[train_end:val_end])].copy()
    test_df  = trainable_df[trainable_df["timestamp"].isin(unique_times[val_end:])].copy()

    # --- Feature columns ---
    drop_cols = [
        "timestamp","symbol","name","trade_date",
        "future_price","future_return","target_class","target_up_binary","training_usable"
    ]
    feature_cols = [
        c for c in train_df.columns
        if c not in drop_cols
        and pd.api.types.is_numeric_dtype(train_df[c])
        and "future" not in c.lower()
        and "target" not in c.lower()
    ]

    # --- Scaling ---
    scaler = StandardScaler()
    train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
    val_df[feature_cols]   = scaler.transform(val_df[feature_cols])
    test_df[feature_cols]  = scaler.transform(test_df[feature_cols])

    # --- Save preprocessed files ---
    train_df.to_csv(os.path.join(PREPROCESSED_DIR, "train_preprocessed.csv"), index=False)
    val_df.to_csv(os.path.join(PREPROCESSED_DIR, "val_preprocessed.csv"), index=False)
    test_df.to_csv(os.path.join(PREPROCESSED_DIR, "test_preprocessed.csv"), index=False)
    pd.DataFrame({"feature": feature_cols}).to_csv(os.path.join(PREPROCESSED_DIR, "feature_columns.csv"), index=False)
    joblib.dump(scaler, os.path.join(PREPROCESSED_DIR, "scaler.pkl"))

    print(f"Train: {train_df.shape}, Val: {val_df.shape}, Test: {test_df.shape}")
    print(f"Features: {len(feature_cols)}")
    print("✅ Preprocessing done.\n")

    return train_df, val_df, test_df, feature_cols


# =============================================================================
# SECTION 5: TRAINING (Binary BUY/SELL with SMOTE)
# =============================================================================

def run_training(train_df: pd.DataFrame, val_df: pd.DataFrame,
                 test_df: pd.DataFrame, feature_cols: List[str]) -> None:
    from imblearn.over_sampling import SMOTE
    from xgboost import XGBClassifier
    from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

    print("\n" + "="*60)
    print("STEP 3: TRAINING (BUY vs SELL - Binary)")
    print("="*60)

    X_train = train_df[feature_cols].copy()
    y_train = train_df["target_class"].astype(int).copy()
    X_val   = val_df[feature_cols].copy()
    y_val   = val_df["target_class"].astype(int).copy()
    X_test  = test_df[feature_cols].copy()
    y_test  = test_df["target_class"].astype(int).copy()

    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    print(f"Train label dist:\n{y_train.value_counts()}")

    # --- Fill NaNs before SMOTE (two-step: median then 0 for all-NaN cols) ---
    print(f"\nNaN count in X_train before fill: {X_train.isna().sum().sum()}")
    train_medians = X_train.median()          # compute once
    X_train = X_train.fillna(train_medians).fillna(0)
    X_val   = X_val.fillna(train_medians).fillna(0)
    X_test  = X_test.fillna(train_medians).fillna(0)
    print(f"NaNs after fill: {X_train.isna().sum().sum()}")

    # --- SMOTE ---
    print("Applying SMOTE...")
    smote = SMOTE(random_state=42)
    X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
    print(f"After SMOTE: {X_train_bal.shape}")

    # --- Sample weights ---
    class_counts = pd.Series(y_train_bal).value_counts().sort_index().to_dict()
    n_total = len(y_train_bal)
    n_classes = len(class_counts)
    class_weight_dict = {cls: n_total / (n_classes * cnt) for cls, cnt in class_counts.items()}
    sample_weights = pd.Series(y_train_bal).map(class_weight_dict).values
    print("Class weights:", class_weight_dict)

    # --- Model ---
    model = XGBClassifier(
        objective="binary:logistic",   # Binary classification
        n_estimators=700,
        max_depth=8,
        learning_rate=0.03,
        subsample=0.90,
        colsample_bytree=0.90,
        min_child_weight=5,
        reg_lambda=1.5,
        reg_alpha=0.2,
        random_state=42,
        n_jobs=-1,
        eval_metric="logloss",
        tree_method="hist"
    )

    print("\nTraining XGBoost (binary)...")
    model.fit(
        X_train_bal, y_train_bal,
        sample_weight=sample_weights,
        eval_set=[(X_val, y_val)],
        verbose=50
    )

    # --- Validation ---
    print("\nValidation Results:")
    val_probs = model.predict_proba(X_val)[:, 1]
    val_pred  = (val_probs >= 0.5).astype(int)

    val_acc  = accuracy_score(y_val, val_pred)
    val_f1_w = f1_score(y_val, val_pred, average="weighted")
    val_f1_m = f1_score(y_val, val_pred, average="macro")
    print(f"Accuracy: {round(val_acc, 4)}")
    print(f"F1 Weighted: {round(val_f1_w, 4)}")
    print(f"F1 Macro: {round(val_f1_m, 4)}")
    print(classification_report(y_val, val_pred, target_names=["SELL", "BUY"], digits=4))
    print("Confusion Matrix:\n", confusion_matrix(y_val, val_pred))

    # --- Val predictions save ---
    val_out = val_df[["symbol", "timestamp"]].copy() if all(c in val_df.columns for c in ["symbol","timestamp"]) else pd.DataFrame()
    val_out["actual_class"]  = y_val.values
    val_out["pred_class"]    = val_pred
    val_out["actual_label"]  = val_out["actual_class"].map(LABEL_MAP)
    val_out["pred_label"]    = val_out["pred_class"].map(LABEL_MAP)
    val_out["prob_sell"]     = 1 - val_probs
    val_out["prob_buy"]      = val_probs
    val_out.to_csv(os.path.join(MODEL_DIR, "val_predictions_v3.csv"), index=False)

    # --- Final model on train + val ---
    print("\nRefitting on train + val combined...")
    X_full = pd.concat([X_train, X_val], axis=0).reset_index(drop=True)
    y_full = pd.concat([y_train, y_val], axis=0).reset_index(drop=True)

    class_counts_full = y_full.value_counts().sort_index().to_dict()
    n_total_full = len(y_full)
    class_weight_full = {cls: n_total_full / (n_classes * cnt) for cls, cnt in class_counts_full.items()}
    sw_full = y_full.map(class_weight_full).values

    final_model = XGBClassifier(
        objective="binary:logistic",
        n_estimators=700,
        max_depth=8,
        learning_rate=0.03,
        subsample=0.90,
        colsample_bytree=0.90,
        min_child_weight=5,
        reg_lambda=1.5,
        reg_alpha=0.2,
        random_state=42,
        n_jobs=-1,
        eval_metric="logloss",
        tree_method="hist"
    )
    final_model.fit(X_full, y_full, sample_weight=sw_full, verbose=False)

    # --- Test evaluation ---
    print("\nTest Results:")
    test_probs = final_model.predict_proba(X_test)[:, 1]
    test_pred  = (test_probs >= 0.5).astype(int)

    test_acc  = accuracy_score(y_test, test_pred)
    test_f1_w = f1_score(y_test, test_pred, average="weighted")
    test_f1_m = f1_score(y_test, test_pred, average="macro")
    print(f"Accuracy: {round(test_acc, 4)}")
    print(f"F1 Weighted: {round(test_f1_w, 4)}")
    print(f"F1 Macro: {round(test_f1_m, 4)}")
    print(classification_report(y_test, test_pred, target_names=["SELL", "BUY"], digits=4))
    print("Confusion Matrix:\n", confusion_matrix(y_test, test_pred))

    # --- Test predictions save ---
    test_out = test_df[["symbol", "timestamp"]].copy() if all(c in test_df.columns for c in ["symbol","timestamp"]) else pd.DataFrame()
    test_out["actual_class"] = y_test.values
    test_out["pred_class"]   = test_pred
    test_out["actual_label"] = test_out["actual_class"].map(LABEL_MAP)
    test_out["pred_label"]   = test_out["pred_class"].map(LABEL_MAP)
    test_out["prob_sell"]    = 1 - test_probs
    test_out["prob_buy"]     = test_probs
    test_out.to_csv(os.path.join(MODEL_DIR, "test_predictions_v3.csv"), index=False)

    # --- Save model + artifacts ---
    model_file      = os.path.join(MODEL_DIR, "xgb_buy_sell_v3.pkl")
    metrics_file    = os.path.join(MODEL_DIR, "xgb_metrics_v3.json")
    importance_file = os.path.join(MODEL_DIR, "xgb_feature_importance_v3.csv")

    joblib.dump(final_model, model_file)

    importance_df = pd.DataFrame({
        "feature": feature_cols,
        "importance": final_model.feature_importances_
    }).sort_values("importance", ascending=False)
    importance_df.to_csv(importance_file, index=False)

    metrics = {
        "label_map": LABEL_MAP,
        "classification": "binary (BUY vs SELL, HOLD removed)",
        "validation": {
            "accuracy": float(val_acc),
            "f1_weighted": float(val_f1_w),
            "f1_macro": float(val_f1_m)
        },
        "test": {
            "accuracy": float(test_acc),
            "f1_weighted": float(test_f1_w),
            "f1_macro": float(test_f1_m)
        },
    }
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n✅ Training complete!")
    print(f"  Model    : {model_file}")
    print(f"  Metrics  : {metrics_file}")
    print(f"  Importance: {importance_file}")


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="PSX Pipeline: Scrape → Preprocess → Train (BUY/SELL only)"
    )
    parser.add_argument(
        "--mode",
        choices=["scrape", "train", "all"],
        default="all",
        help=(
            "scrape = only scrape data\n"
            "train  = preprocess + train only (uses existing CSV)\n"
            "all    = scrape + preprocess + train\n"
            "(default: all)"
        )
    )
    parser.add_argument(
        "--input-csv",
        default=ALL_CONS_CSV,
        help=f"Preprocessing ke liye CSV path (default: {ALL_CONS_CSV})"
    )
    args = parser.parse_args()

    print("\n" + "="*60)
    print("PSX MERGED PIPELINE")
    print(f"Mode: {args.mode.upper()}")
    print("="*60)

    # =========================================================
    # MARKET HOURS SCHEDULER
    # PSX Market: Monday-Friday  09:30 - 15:30 PKT (UTC+5)
    # =========================================================
    import time as _time
    from datetime import datetime, timezone, timedelta

    PKT = timezone(timedelta(hours=5))
    MARKET_OPEN         = (9,  30)   # 09:30 PKT
    MARKET_CLOSE        = (15, 30)   # 15:30 PKT
    SCRAPE_INTERVAL_SEC = 300        # scrape every 5 minutes
    MIN_TIMESTAMPS      = 10         # collect at least 10 snapshots before training

    def pkt_now():
        return datetime.now(PKT)

    def is_market_open():
        now = pkt_now()
        if now.weekday() >= 5:       # Saturday or Sunday
            return False
        t = (now.hour, now.minute)
        return MARKET_OPEN <= t < MARKET_CLOSE

    def seconds_until_market_open():
        now = pkt_now()
        days_ahead = 0
        candidate = now
        while True:
            if candidate.weekday() < 5:
                open_time = candidate.replace(
                    hour=MARKET_OPEN[0], minute=MARKET_OPEN[1], second=0, microsecond=0
                )
                if open_time > now:
                    return (open_time - now).total_seconds()
            days_ahead += 1
            candidate = now + timedelta(days=days_ahead)
            candidate = candidate.replace(
                hour=MARKET_OPEN[0], minute=MARKET_OPEN[1], second=0, microsecond=0
            )
            if candidate.weekday() < 5:
                return (candidate - now).total_seconds()

    def get_timestamp_count():
        if not os.path.exists(ALL_CONS_CSV):
            return 0
        try:
            tmp = pd.read_csv(ALL_CONS_CSV, engine="python", on_bad_lines="skip")
            return tmp["ScrapeTimestamp"].nunique()
        except Exception:
            return 0

    def get_today_timestamp_count():
        """Count how many scrape rounds have happened TODAY specifically."""
        if not os.path.exists(ALL_CONS_CSV):
            return 0
        try:
            tmp = pd.read_csv(ALL_CONS_CSV, engine="python", on_bad_lines="skip")
            today = pkt_now().strftime("%Y-%m-%d")
            today_rows = tmp[tmp["ScrapeTimestamp"].str.startswith(today)]
            return today_rows["ScrapeTimestamp"].nunique()
        except Exception:
            return 0

    # How many fresh rounds to scrape TODAY (regardless of historical data)
    TARGET_ROUNDS_TODAY = 10

    if args.mode in ("scrape", "all"):
        print("\nSTEP 1: SCRAPING (Market Hours Scheduler)")
        print("="*60)
        print(f"PSX Market Hours     : Mon-Fri  09:30 - 15:30 PKT")
        print(f"Scrape Interval      : every {SCRAPE_INTERVAL_SEC//60} minutes")
        print(f"Target rounds today  : {TARGET_ROUNDS_TODAY}")
        print(f"Total snapshots so far: {get_timestamp_count()}")

        round_num = 1
        while get_today_timestamp_count() < TARGET_ROUNDS_TODAY:
            now_pkt  = pkt_now()
            ts_count = get_timestamp_count()

            print(f"\n[{now_pkt.strftime('%Y-%m-%d %H:%M:%S PKT')}]  "
                  f"Snapshots: {ts_count}/{MIN_TIMESTAMPS}")

            if not is_market_open():
                wait_sec  = seconds_until_market_open()
                wait_hr   = wait_sec / 3600
                next_open = pkt_now() + timedelta(seconds=wait_sec)
                print(f"⏸  Market is closed  —  "
                      f"{now_pkt.strftime('%A')} {now_pkt.strftime('%H:%M PKT')}")
                print(f"⏰ Next open : {next_open.strftime('%A %d %b %Y — %H:%M PKT')}"
                      f"  ({wait_hr:.1f} hours away)")
                print("   Sleeping until market opens...")
                _time.sleep(min(wait_sec, 3600))   # re-check every hour at most
                continue

            print(f"✅ Market is open — starting Scrape Round {round_num} ...")
            scrape_all_index_constituents()
            round_num += 1

            ts_count = get_today_timestamp_count()
            print(f"📊 Rounds today : {ts_count} / {TARGET_ROUNDS_TODAY}  |  Total snapshots: {get_timestamp_count()}")

            if ts_count < TARGET_ROUNDS_TODAY:
                # Wait for the next scrape interval; break early if market closes
                wait_end = _time.time() + SCRAPE_INTERVAL_SEC
                while _time.time() < wait_end:
                    if not is_market_open():
                        print("⏸  Market has closed — pausing until next session.")
                        break
                    remaining = int(wait_end - _time.time())
                    print(f"   Next scrape in {remaining}s ...", end="\r")
                    _time.sleep(30)

        scrape_listings()
        print(f"\n✅ Today: {get_today_timestamp_count()} rounds done  |  Total snapshots: {get_timestamp_count()}")

    if args.mode in ("train", "all"):
        input_csv = args.input_csv if args.mode == "train" else ALL_CONS_CSV
        if not os.path.exists(input_csv):
            print(f"❌ Input CSV not found: {input_csv}")
            print("Run --mode scrape first, or provide --input-csv path.")
            sys.exit(1)

        train_df, val_df, test_df, feature_cols = run_preprocessing(input_csv)
        run_training(train_df, val_df, test_df, feature_cols)

    print("\n✅ Pipeline complete!")
    print(f"Output folder: {OUTPUT_DIR}")
def predict(symbol):
    MODEL_PATH = os.path.join(
        BASE_DIR,
        "output",
        "model_output_xgb_v3",
        "xgb_buy_sell_v3.pkl"
    )

    model = joblib.load(MODEL_PATH)

    features = np.random.rand(1, 27)

    prob_buy = model.predict_proba(features)[0][1]
    prediction = "BUY" if prob_buy >= 0.5 else "SELL"

    return prediction, float(prob_buy)
# =========================
# MODEL PREDICTION FUNCTION
# =========================
def predict(symbol):

    MODEL_PATH = os.path.join(
        BASE_DIR,
        "output",
        "model_output_xgb_v3",
        "xgb_buy_sell_v3.pkl"
    )

    model = joblib.load(MODEL_PATH)

    # Temporary random features
    features = np.random.rand(1, 27)

    prob_buy = model.predict_proba(features)[0][1]

    prediction = "BUY" if prob_buy >= 0.5 else "SELL"

    return prediction, float(prob_buy)
if __name__ == "__main__":
    main()

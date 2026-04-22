"""
Sovereign Bond Yield & Spread Panel Builder
============================================

Builds a clean country-quarter panel of long-term sovereign bond yields,
spreads vs. benchmarks (US Treasury and German Bund), macro controls, and
selected climate indicators — ready to merge with climate-transition
variables for dissertation research.

Data sources (all free):
    * FRED (Federal Reserve Economic Data)  — primary yield feed
        https://fred.stlouisfed.org/   (API key required — free signup)
    * OECD SDMX                              — yield fallback for non-FRED members
        https://sdmx.oecd.org/public/rest/data/
    * World Bank WDI                         — macro controls & CO2 variables
        https://data.worldbank.org/         (no key needed)
    * BIS (Bank for International Settlements)  — government debt stats
        https://stats.bis.org/                (public CSV / SDMX)

Output:
    outputs/sovereign_panel_wide.csv     one row per country-quarter
    outputs/sovereign_panel_long.csv     tidy long format (country,quarter,var,value)
    outputs/sovereign_panel.parquet      same wide frame in parquet for analysis

Usage:
    export FRED_API_KEY="<your_free_key>"       # required
    pip install -r requirements-panel.txt
    python sovereign_panel_builder.py

Author: Prof. V. Ravichandran | The Mountain Path Academy
"""
from __future__ import annotations

import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# =============================================================================
# 0. CONFIGURATION
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("panel")

# ---------- date range ----------
START = "2005-01-01"   # post-EMU, covers GFC, Euro crisis, COVID, and ESG-era
END   = None           # None → up to latest available

# ---------- output directory ----------
OUT_DIR = Path("./outputs_panel")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------- API keys ----------
FRED_API_KEY = os.getenv("FRED_API_KEY")

# =============================================================================
# 1. COUNTRY UNIVERSE
#
# 40 countries split into Advanced Economies (AE) and Emerging Markets (EM).
# ISO3 + ISO2 codes are carried everywhere so merges are unambiguous.
# The `fred_ltir` column gives the FRED ticker for the OECD-harmonised
# long-term interest rate (10-year government benchmark, monthly, %).
# Where FRED does not publish a series, set to None and the script will
# attempt an OECD SDMX fallback.
# =============================================================================
@dataclass
class Country:
    iso3: str
    iso2: str
    name: str
    group: str              # "AE" or "EM"
    fred_ltir: Optional[str]   # FRED ticker, if available

COUNTRIES: List[Country] = [
    # ---- Advanced Economies ----
    Country("USA", "US", "United States",  "AE", "IRLTLT01USM156N"),
    Country("GBR", "GB", "United Kingdom", "AE", "IRLTLT01GBM156N"),
    Country("DEU", "DE", "Germany",        "AE", "IRLTLT01DEM156N"),
    Country("FRA", "FR", "France",         "AE", "IRLTLT01FRM156N"),
    Country("ITA", "IT", "Italy",          "AE", "IRLTLT01ITM156N"),
    Country("ESP", "ES", "Spain",          "AE", "IRLTLT01ESM156N"),
    Country("NLD", "NL", "Netherlands",    "AE", "IRLTLT01NLM156N"),
    Country("BEL", "BE", "Belgium",        "AE", "IRLTLT01BEM156N"),
    Country("AUT", "AT", "Austria",        "AE", "IRLTLT01ATM156N"),
    Country("IRL", "IE", "Ireland",        "AE", "IRLTLT01IEM156N"),
    Country("PRT", "PT", "Portugal",       "AE", "IRLTLT01PTM156N"),
    Country("GRC", "GR", "Greece",         "AE", "IRLTLT01GRM156N"),
    Country("FIN", "FI", "Finland",        "AE", "IRLTLT01FIM156N"),
    Country("SWE", "SE", "Sweden",         "AE", "IRLTLT01SEM156N"),
    Country("DNK", "DK", "Denmark",        "AE", "IRLTLT01DKM156N"),
    Country("NOR", "NO", "Norway",         "AE", "IRLTLT01NOM156N"),
    Country("CHE", "CH", "Switzerland",    "AE", "IRLTLT01CHM156N"),
    Country("CAN", "CA", "Canada",         "AE", "IRLTLT01CAM156N"),
    Country("AUS", "AU", "Australia",      "AE", "IRLTLT01AUM156N"),
    Country("NZL", "NZ", "New Zealand",    "AE", "IRLTLT01NZM156N"),
    Country("JPN", "JP", "Japan",          "AE", "IRLTLT01JPM156N"),
    Country("KOR", "KR", "Korea",          "AE", "IRLTLT01KRM156N"),
    Country("ISR", "IL", "Israel",         "AE", None),               # OECD fallback
    # ---- Emerging Markets ----
    Country("MEX", "MX", "Mexico",         "EM", "IRLTLT01MXM156N"),
    Country("CHL", "CL", "Chile",          "EM", "IRLTLT01CLM156N"),
    Country("COL", "CO", "Colombia",       "EM", "IRLTLT01COM156N"),
    Country("PER", "PE", "Peru",           "EM", None),               # OECD fallback
    Country("BRA", "BR", "Brazil",         "EM", None),               # Bloomberg/Refinitiv
    Country("ZAF", "ZA", "South Africa",   "EM", "IRLTLT01ZAM156N"),
    Country("TUR", "TR", "Turkey",         "EM", "IRLTLT01TRM156N"),
    Country("POL", "PL", "Poland",         "EM", "IRLTLT01PLM156N"),
    Country("CZE", "CZ", "Czech Republic", "EM", "IRLTLT01CZM156N"),
    Country("HUN", "HU", "Hungary",        "EM", "IRLTLT01HUM156N"),
    Country("ROU", "RO", "Romania",        "EM", None),               # OECD fallback
    Country("IND", "IN", "India",          "EM", "INDIRLTLT01STM"),
    Country("IDN", "ID", "Indonesia",      "EM", "IDNIRLTLT01STM"),
    Country("MYS", "MY", "Malaysia",       "EM", None),               # Bloomberg/Refinitiv
    Country("PHL", "PH", "Philippines",    "EM", None),               # Bloomberg/Refinitiv
    Country("THA", "TH", "Thailand",       "EM", None),               # Bloomberg/Refinitiv
    Country("CHN", "CN", "China",          "EM", "INTGSTCNM193N"),    # PBoC short, proxy
    Country("EGY", "EG", "Egypt",          "EM", None),               # Bloomberg/Refinitiv
]

# Benchmarks used for spread construction
BENCHMARK_US = "USA"
BENCHMARK_DE = "DEU"

# =============================================================================
# 2. FRED LOADER — long-term interest rates
# =============================================================================
def load_fred_yields() -> pd.DataFrame:
    """Pull long-term (10Y benchmark) yields for every country that has a FRED
    ticker. Returns a monthly wide frame indexed by date, columns = ISO3."""
    if not FRED_API_KEY:
        log.error("FRED_API_KEY not set — export it before running.")
        sys.exit(1)

    try:
        from fredapi import Fred
    except ImportError:
        log.error("Missing dependency: pip install fredapi")
        sys.exit(1)

    fred = Fred(api_key=FRED_API_KEY)
    series_map = {c.iso3: c.fred_ltir for c in COUNTRIES if c.fred_ltir}
    log.info(f"Requesting {len(series_map)} FRED yield series ...")

    frames = {}
    for iso3, ticker in series_map.items():
        try:
            s = fred.get_series(ticker,
                                observation_start=START,
                                observation_end=END)
            frames[iso3] = s.rename(iso3)
            log.info(f"   ✔  {iso3:<3} ({ticker})  {len(s):>4} obs")
            time.sleep(0.15)           # be polite to the API
        except Exception as e:
            log.warning(f"   ✖  {iso3} ({ticker}) failed: {e}")

    if not frames:
        raise RuntimeError("No FRED series returned — check your API key.")

    df = pd.concat(frames.values(), axis=1).sort_index()
    df.index = pd.to_datetime(df.index)
    df.index.name = "date"
    return df


# =============================================================================
# 3. OECD SDMX FALLBACK — for countries without FRED tickers
# =============================================================================
def load_oecd_yields(missing_iso2: List[str]) -> pd.DataFrame:
    """Pull long-term interest rates from the OECD SDMX API for countries not
    covered by FRED. Returns monthly wide frame (columns = ISO3)."""
    if not missing_iso2:
        log.info("No OECD fallback needed.")
        return pd.DataFrame()

    try:
        import pandasdmx as sdmx
    except ImportError:
        log.warning("pandasdmx not installed — skipping OECD fallback. "
                    "Install with:  pip install pandasdmx")
        return pd.DataFrame()

    iso2_to_iso3 = {c.iso2: c.iso3 for c in COUNTRIES}
    oecd = sdmx.Request("OECD")
    # "KEI" (Key Economic Indicators), IRLT = long-term interest rates,
    # ST = stock (rate), M = monthly
    # Key pattern: {country}.IRLT.ST.M
    frames = {}
    for iso2 in missing_iso2:
        try:
            key = f"{iso2}.IRLT.ST.M"
            resp = oecd.data(resource_id="KEI", key=key,
                             params=dict(startTime=START[:7]))
            ser = resp.to_pandas()
            if ser.empty:
                raise ValueError("empty series")
            # pandasdmx returns a multiindex; flatten to simple monthly series
            if isinstance(ser.index, pd.MultiIndex):
                ser = ser.reset_index(level=list(range(ser.index.nlevels - 1)),
                                      drop=True)
            ser.index = pd.to_datetime(ser.index)
            iso3 = iso2_to_iso3[iso2]
            frames[iso3] = ser.rename(iso3)
            log.info(f"   ✔  OECD {iso3} ({iso2})  {len(ser):>4} obs")
            time.sleep(0.2)
        except Exception as e:
            log.warning(f"   ✖  OECD {iso2} failed: {e}")

    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames.values(), axis=1).sort_index()
    df.index.name = "date"
    return df


# =============================================================================
# 4. WORLD BANK MACRO + CLIMATE CONTROLS (annual)
# =============================================================================
WB_INDICATORS = {
    # --- macro fundamentals ---
    "gdp_usd":      "NY.GDP.MKTP.CD",           # GDP current US$
    "gdp_growth":   "NY.GDP.MKTP.KD.ZG",        # GDP growth, real %
    "cpi_yoy":      "FP.CPI.TOTL.ZG",           # CPI inflation, %
    "debt_gdp":     "GC.DOD.TOTL.GD.ZS",        # central gov debt % GDP
    "cab_gdp":      "BN.CAB.XOKA.GD.ZS",        # current account % GDP
    "reserves_mo":  "FI.RES.TOTL.MO",           # reserves, months of imports
    "trade_gdp":    "NE.TRD.GNFS.ZS",           # trade openness % GDP
    # --- climate / carbon ---
    "co2_kt":       "EN.ATM.CO2E.KT",           # total CO2 emissions, kt
    "co2_pc":       "EN.ATM.CO2E.PC",           # CO2 per capita, t
    "co2_gdp":      "EN.ATM.CO2E.KD.GD",        # kg CO2 per 2015 USD GDP
    "renew_share":  "EG.FEC.RNEW.ZS",           # renewables % final energy
    "energy_gdp":   "EG.USE.COMM.GD.PP.KD",     # energy intensity
}

def load_worldbank_macro() -> pd.DataFrame:
    """Pull annual World Bank indicators for every ISO3. Returns a long frame
    with columns [iso3, year, <indicator>..., ...]."""
    try:
        import wbdata
    except ImportError:
        log.warning("wbdata not installed — skipping World Bank. "
                    "Install with:  pip install wbdata")
        return pd.DataFrame()

    import datetime as dt
    iso3_list = [c.iso3 for c in COUNTRIES]
    # wbdata signature changed in recent versions — use inclusive date range
    start_year = int(START[:4])
    end_year   = dt.datetime.today().year
    data_date  = (dt.datetime(start_year, 1, 1), dt.datetime(end_year, 12, 31))

    log.info(f"Requesting {len(WB_INDICATORS)} WB indicators "
             f"for {len(iso3_list)} countries ...")
    try:
        df = wbdata.get_dataframe(
            indicators=WB_INDICATORS,
            country=iso3_list,
            date=data_date,
        )
    except TypeError:
        # older wbdata used data_date=
        df = wbdata.get_dataframe(
            indicators=WB_INDICATORS,
            country=iso3_list,
            data_date=data_date,
        )
    df = df.reset_index()
    # normalise column names
    df = df.rename(columns={"country": "country_name", "date": "year"})
    # wbdata returns date as string like '2019' — coerce
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    # map country_name back to iso3
    name_to_iso3 = {c.name: c.iso3 for c in COUNTRIES}
    df["iso3"] = df["country_name"].map(name_to_iso3)
    missing = df.loc[df["iso3"].isna(), "country_name"].unique()
    if len(missing):
        log.warning(f"WB country-name mismatch: {missing}. "
                    "Update COUNTRIES.name to the official WB name.")
    df = df.dropna(subset=["iso3", "year"]).drop(columns="country_name")
    col_order = ["iso3", "year"] + list(WB_INDICATORS.keys())
    return df[col_order].sort_values(["iso3", "year"]).reset_index(drop=True)


# =============================================================================
# 5. BIS — government debt securities (quarterly)
# =============================================================================
# BIS publishes a flat CSV of "Credit to non-financial sectors" which includes
# credit to general government at market value.  The public URL is stable.
BIS_CREDIT_CSV = (
    "https://data.bis.org/static/bulk/WS_TC_csv_flat.zip"
)

def load_bis_gov_debt() -> pd.DataFrame:
    """Best-effort BIS pull.  Returns long frame [iso3, quarter, bis_gov_debt_pct_gdp].
    Returns empty DataFrame on failure — BIS occasionally moves files."""
    try:
        import io, zipfile, requests
    except ImportError:
        return pd.DataFrame()

    iso3_set = {c.iso3 for c in COUNTRIES}
    try:
        log.info("Downloading BIS total-credit flat file ...")
        r = requests.get(BIS_CREDIT_CSV, timeout=60)
        r.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
            fname = [n for n in zf.namelist() if n.endswith(".csv")][0]
            with zf.open(fname) as f:
                df = pd.read_csv(f, low_memory=False)
        log.info(f"   ✔  BIS rows: {len(df):,}")
    except Exception as e:
        log.warning(f"BIS download failed: {e} — skipping BIS block.")
        return pd.DataFrame()

    # BIS schema: BORROWERS_CTY, TC_BORROWERS (sector), UNIT_TYPE, etc.
    # We want: government sector, pct of GDP, at market value.
    #   TC_BORROWERS == "G"   (general government)
    #   UNIT_TYPE    == "770" (% of GDP, market value)
    try:
        filt = (
            df.get("TC_BORROWERS").astype(str).str.upper().eq("G")
            & df.get("UNIT_TYPE").astype(str).eq("770")
            & df.get("BORROWERS_CTY").isin(iso3_set)
        )
        sub = df.loc[filt, ["BORROWERS_CTY", "TIME_PERIOD", "OBS_VALUE"]].copy()
        sub.columns = ["iso3", "quarter", "bis_gov_debt_pct_gdp"]
        sub["quarter"] = pd.PeriodIndex(sub["quarter"], freq="Q")
        sub["bis_gov_debt_pct_gdp"] = pd.to_numeric(
            sub["bis_gov_debt_pct_gdp"], errors="coerce"
        )
        sub = sub.dropna().sort_values(["iso3", "quarter"]).reset_index(drop=True)
        log.info(f"   ✔  BIS filtered rows: {len(sub):,}")
        return sub
    except Exception as e:
        log.warning(f"BIS filter failed: {e} — returning empty frame.")
        return pd.DataFrame()


# =============================================================================
# 6. HARMONISATION — monthly yields → quarterly panel
# =============================================================================
def monthly_to_quarterly(wide_monthly: pd.DataFrame) -> pd.DataFrame:
    """Aggregate monthly series to quarter means.  Returns long frame:
    [iso3, quarter, yield_10y]."""
    if wide_monthly.empty:
        return pd.DataFrame(columns=["iso3", "quarter", "yield_10y"])
    # pandas 2.2+ deprecated 'Q' in favour of 'QE'. Handle both gracefully.
    try:
        q = wide_monthly.resample("QE").mean()
    except ValueError:
        q = wide_monthly.resample("Q").mean()   # pandas < 2.2
    q = q.rename_axis("date")
    q.index = q.index.to_period("Q")
    long = (
        q.stack().rename("yield_10y")
         .reset_index()
         .rename(columns={"date": "quarter", "level_1": "iso3"})
    )
    long = long[["iso3", "quarter", "yield_10y"]]
    return long


def build_spreads(yields_long: pd.DataFrame) -> pd.DataFrame:
    """Add spread_vs_us and spread_vs_de (in basis points)."""
    if yields_long.empty:
        return yields_long

    us = (
        yields_long.query("iso3 == @BENCHMARK_US")
        .set_index("quarter")["yield_10y"]
        .rename("us_y")
    )
    de = (
        yields_long.query("iso3 == @BENCHMARK_DE")
        .set_index("quarter")["yield_10y"]
        .rename("de_y")
    )
    out = yields_long.set_index("quarter").join([us, de])
    out["spread_vs_us_bp"] = (out["yield_10y"] - out["us_y"]) * 100
    out["spread_vs_de_bp"] = (out["yield_10y"] - out["de_y"]) * 100
    out = out.reset_index().drop(columns=["us_y", "de_y"])
    return out[["iso3", "quarter", "yield_10y",
                "spread_vs_us_bp", "spread_vs_de_bp"]]


def annual_to_quarterly(annual_long: pd.DataFrame,
                        id_cols=("iso3", "year")) -> pd.DataFrame:
    """Expand annual macro to quarterly by forward-filling within the year.
    Returns long frame with a 'quarter' column replacing 'year'."""
    if annual_long.empty:
        return annual_long
    rows = []
    val_cols = [c for c in annual_long.columns if c not in id_cols]
    for _, r in annual_long.iterrows():
        y = int(r["year"])
        for q in range(1, 5):
            rec = {"iso3": r["iso3"],
                   "quarter": pd.Period(f"{y}Q{q}", freq="Q")}
            rec.update({c: r[c] for c in val_cols})
            rows.append(rec)
    return pd.DataFrame(rows)


# =============================================================================
# 7. MAIN ASSEMBLY
# =============================================================================
def build_panel() -> pd.DataFrame:
    # 7.1 Yields from FRED
    fred_wide = load_fred_yields()

    # 7.2 OECD fallback for countries without FRED ticker
    missing = [c.iso2 for c in COUNTRIES
               if (c.fred_ltir is None) or (c.iso3 not in fred_wide.columns)]
    oecd_wide = load_oecd_yields(missing)

    # combine (prefer FRED where overlap)
    yields_wide = fred_wide.combine_first(oecd_wide) \
        if not oecd_wide.empty else fred_wide
    log.info(f"Yield panel: {yields_wide.shape[1]} countries, "
             f"{yields_wide.shape[0]} monthly rows.")

    # 7.3 Quarterly aggregation & spreads
    yields_q = monthly_to_quarterly(yields_wide)
    yields_q = build_spreads(yields_q)

    # 7.4 World Bank macro + climate (annual → quarterly)
    wb_annual = load_worldbank_macro()
    wb_q = annual_to_quarterly(wb_annual)

    # 7.5 BIS quarterly government debt
    bis_q = load_bis_gov_debt()

    # 7.6 Merge everything on (iso3, quarter)
    panel = yields_q.copy()
    if not wb_q.empty:
        panel = panel.merge(wb_q, on=["iso3", "quarter"], how="left")
    if not bis_q.empty:
        panel = panel.merge(bis_q, on=["iso3", "quarter"], how="left")

    # 7.7 Attach country metadata (name, group)
    meta = pd.DataFrame(
        [(c.iso3, c.iso2, c.name, c.group) for c in COUNTRIES],
        columns=["iso3", "iso2", "country_name", "group"],
    )
    panel = panel.merge(meta, on="iso3", how="left")

    # 7.8 Tidy column order
    lead = ["iso3", "iso2", "country_name", "group", "quarter",
            "yield_10y", "spread_vs_us_bp", "spread_vs_de_bp"]
    other = [c for c in panel.columns if c not in lead]
    panel = panel[lead + other].sort_values(["iso3", "quarter"]) \
                                .reset_index(drop=True)
    return panel


def save_outputs(panel: pd.DataFrame) -> None:
    wide_path   = OUT_DIR / "sovereign_panel_wide.csv"
    long_path   = OUT_DIR / "sovereign_panel_long.csv"
    parq_path   = OUT_DIR / "sovereign_panel.parquet"

    panel.to_csv(wide_path, index=False)
    log.info(f"Wrote {wide_path}  ({len(panel):,} rows × {panel.shape[1]} cols)")

    long = panel.melt(
        id_vars=["iso3", "iso2", "country_name", "group", "quarter"],
        var_name="variable", value_name="value",
    ).dropna(subset=["value"])
    long.to_csv(long_path, index=False)
    log.info(f"Wrote {long_path}  ({len(long):,} rows)")

    try:
        panel.assign(quarter=panel["quarter"].astype(str)) \
             .to_parquet(parq_path, index=False)
        log.info(f"Wrote {parq_path}")
    except Exception as e:
        log.warning(f"parquet skipped: {e}  (pip install pyarrow)")


def summary(panel: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print(" PANEL SUMMARY ".center(70, "="))
    print("=" * 70)
    print(f"Countries : {panel['iso3'].nunique()}")
    print(f"Quarters  : {panel['quarter'].min()}  →  {panel['quarter'].max()}")
    print(f"Rows      : {len(panel):,}")
    print(f"Columns   : {panel.shape[1]}")
    print("\nColumn coverage (% non-null):")
    cov = panel.notna().mean().mul(100).round(1).sort_values(ascending=False)
    print(cov.to_string())
    print("\nPer-country yield obs:")
    print(panel.groupby("iso3")["yield_10y"].count()
                .sort_values(ascending=False).to_string())
    print("=" * 70)


# =============================================================================
# 8. ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    log.info("=== Sovereign Panel Builder — start ===")
    panel = build_panel()
    save_outputs(panel)
    summary(panel)
    log.info("=== Done. Merge climate-transition variables on "
             "(iso3, quarter) to complete your dissertation dataset. ===")

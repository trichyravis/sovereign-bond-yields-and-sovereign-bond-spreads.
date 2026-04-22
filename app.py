
"""
Sovereign Bond Yield & Spread Panel Builder — Streamlit App
============================================================
The Mountain Path Academy · World of Finance

An interactive web tool that assembles a country-quarter panel of long-term
sovereign bond yields, spreads vs. US Treasury and German Bund, macro controls,
and selected climate indicators — ready to merge with climate-transition
variables for dissertation research on climate & sovereign debt.

User enters their FRED API key (password input — never stored), clicks a button,
and downloads the resulting CSV / Parquet.

Run:
    pip install streamlit pandas numpy fredapi wbdata pandasdmx requests plotly pyarrow
    streamlit run panel_builder_app.py
"""
from __future__ import annotations

import io
import time
import zipfile
from dataclasses import dataclass
from datetime import date, datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# =========================================================================
# PAGE CONFIG
# =========================================================================
st.set_page_config(
    page_title="Sovereign Panel Builder | The Mountain Path Academy",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================================================================
# MOUNTAIN PATH ACADEMY DESIGN SYSTEM
# =========================================================================
MPA_NAVY       = "#0E2C54"
MPA_NAVY_DARK  = "#0A2145"
MPA_YELLOW     = "#F4C430"
MPA_LIGHT_BLUE = "#EAF1F8"
MPA_GREEN      = "#2E7D4F"
MPA_LIGHT_GREEN= "#E3F1E8"
MPA_ORANGE     = "#D2691E"
MPA_LIGHT_OR   = "#FFF0DC"
MPA_RED        = "#B02A37"
MPA_LIGHT_RED  = "#FBE5E7"
MPA_LIGHT_YEL  = "#FFF6DD"
MPA_PURPLE     = "#6B46A1"
MPA_LIGHT_PUR  = "#EEE6F7"
MPA_GREY       = "#5A6B7B"
MPA_LIGHT_GREY = "#F2F4F7"

st.markdown(f"""
<style>
    .stApp {{ background: #ffffff; }}
    .block-container {{ padding-top: 1rem; padding-bottom: 3rem; max-width: 1250px; }}
    #MainMenu {{ visibility: hidden; }}
    footer {{ visibility: hidden; }}
    header[data-testid="stHeader"] {{ background: transparent; height: 0; }}
    html, body, [class*="css"] {{
        font-family: "Lato", "Segoe UI", -apple-system, sans-serif;
        color: #212529;
    }}
    h1, h2, h3, h4 {{ color: {MPA_NAVY}; font-weight: 700; }}

    /* top banner */
    .mpa-topbar {{
        background: {MPA_NAVY};
        padding: 18px 28px 14px 28px;
        border-bottom: 4px solid {MPA_YELLOW};
        margin: -1rem -1rem 1.5rem -1rem;
        display: flex; justify-content: space-between; align-items: center;
    }}
    .mpa-topbar .title {{ color: white; font-size: 1.25rem; font-weight: 700; letter-spacing: 0.5px; }}
    .mpa-topbar .sub   {{ color: {MPA_YELLOW}; font-size: 0.85rem; font-style: italic; }}

    /* section headers */
    .mpa-section {{
        color: {MPA_NAVY}; font-size: 1.5rem; font-weight: 800;
        padding-bottom: 6px; margin: 1.2rem 0 1rem 0;
        border-bottom: 2px solid {MPA_YELLOW};
    }}

    /* info box */
    .mpa-box {{
        border-radius: 6px;
        padding: 14px 16px;
        margin: 10px 0;
        border: 1px solid {MPA_NAVY};
        background: {MPA_LIGHT_BLUE};
    }}
    .mpa-box-warn {{
        border-color: {MPA_RED}; background: {MPA_LIGHT_RED};
    }}
    .mpa-box-ok {{
        border-color: {MPA_GREEN}; background: {MPA_LIGHT_GREEN};
    }}
    .mpa-box-insight {{
        border-color: {MPA_YELLOW}; background: {MPA_LIGHT_YEL};
    }}

    /* ============ sidebar ============ */
    section[data-testid="stSidebar"] {{ background: {MPA_NAVY}; }}
    /* base text: light on navy */
    section[data-testid="stSidebar"],
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] div,
    section[data-testid="stSidebar"] .stMarkdown {{
        color: #FFFFFF !important;
    }}
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {{
        color: {MPA_YELLOW} !important;
        border-bottom: 1px solid {MPA_YELLOW};
        padding-bottom: 4px; margin-top: 1rem;
    }}
    /* Captions and help text in soft blue-grey */
    section[data-testid="stSidebar"] .stCaption,
    section[data-testid="stSidebar"] small {{
        color: #BFD3E6 !important;
    }}

    /* ---- Input wrappers (text / password / number) ----
       Streamlit uses BaseWeb <div data-baseweb="input"> for the outer shell
       and a nested <input> for the actual field. Without targeting both,
       the white default background survives and makes white text invisible. */
    section[data-testid="stSidebar"] [data-baseweb="input"],
    section[data-testid="stSidebar"] [data-baseweb="select"] > div,
    section[data-testid="stSidebar"] [data-baseweb="base-input"] {{
        background-color: rgba(255,255,255,0.10) !important;
        border: 1px solid {MPA_YELLOW} !important;
        border-radius: 4px !important;
        color: #FFFFFF !important;
    }}
    section[data-testid="stSidebar"] [data-baseweb="input"] input,
    section[data-testid="stSidebar"] [data-baseweb="base-input"] input,
    section[data-testid="stSidebar"] [data-baseweb="select"] input,
    section[data-testid="stSidebar"] textarea {{
        background: transparent !important;
        color: #FFFFFF !important;
        -webkit-text-fill-color: #FFFFFF !important;  /* Safari password fix */
        caret-color: {MPA_YELLOW} !important;
    }}
    /* placeholder (e.g. "32-character hex string") */
    section[data-testid="stSidebar"] input::placeholder,
    section[data-testid="stSidebar"] textarea::placeholder {{
        color: rgba(255,255,255,0.55) !important;
    }}
    /* Number-input +/- step buttons */
    section[data-testid="stSidebar"] button[data-testid="stNumberInput-StepUp"],
    section[data-testid="stSidebar"] button[data-testid="stNumberInput-StepDown"] {{
        background: rgba(244,196,48,0.15) !important;
        color: {MPA_YELLOW} !important;
        border-color: {MPA_YELLOW} !important;
    }}

    /* ---- Multiselect chips — yellow on navy, readable ---- */
    section[data-testid="stSidebar"] [data-baseweb="tag"] {{
        background-color: {MPA_YELLOW} !important;
        color: {MPA_NAVY_DARK} !important;
        border: none !important;
    }}
    section[data-testid="stSidebar"] [data-baseweb="tag"] span,
    section[data-testid="stSidebar"] [data-baseweb="tag"] div {{
        color: {MPA_NAVY_DARK} !important;
        font-weight: 600 !important;
    }}
    /* the "x" icon inside a chip */
    section[data-testid="stSidebar"] [data-baseweb="tag"] svg {{
        fill: {MPA_NAVY_DARK} !important;
    }}
    /* Dropdown chevron */
    section[data-testid="stSidebar"] [data-baseweb="select"] svg {{
        fill: {MPA_YELLOW} !important;
    }}

    /* ---- Checkbox — yellow tick, white label ---- */
    section[data-testid="stSidebar"] [data-testid="stCheckbox"] label p {{
        color: white !important;
    }}
    section[data-testid="stSidebar"] [data-testid="stCheckbox"] svg {{
        color: {MPA_YELLOW} !important;
        fill: {MPA_YELLOW} !important;
    }}

    /* Links stay yellow */
    section[data-testid="stSidebar"] a {{
        color: {MPA_YELLOW} !important; text-decoration: underline;
    }}

    /* Dropdown menus (appears when opening a selectbox inside sidebar) */
    div[data-baseweb="popover"] ul {{
        background: {MPA_NAVY} !important;
    }}
    div[data-baseweb="popover"] ul li {{
        color: white !important;
    }}
    div[data-baseweb="popover"] ul li:hover {{
        background: {MPA_NAVY_DARK} !important; color: {MPA_YELLOW} !important;
    }}

    /* metric cards */
    div[data-testid="stMetric"] {{
        background: {MPA_LIGHT_BLUE};
        border-left: 4px solid {MPA_NAVY};
        padding: 10px 14px; border-radius: 4px;
    }}
    div[data-testid="stMetric"] label {{ color: {MPA_NAVY}; font-weight: 600; }}

    /* buttons */
    .stButton > button, .stDownloadButton > button {{
        background: {MPA_NAVY}; color: white; border: none;
        border-radius: 4px; font-weight: 600; padding: 8px 20px;
    }}
    .stButton > button:hover, .stDownloadButton > button:hover {{
        background: {MPA_YELLOW}; color: {MPA_NAVY_DARK};
    }}

    /* tabs */
    .stTabs [data-baseweb="tab-list"] {{ gap: 4px; border-bottom: 2px solid {MPA_YELLOW}; }}
    .stTabs [data-baseweb="tab"] {{
        background: {MPA_LIGHT_BLUE}; color: {MPA_NAVY};
        border-radius: 4px 4px 0 0; padding: 8px 18px; font-weight: 600;
    }}
    .stTabs [aria-selected="true"] {{
        background: {MPA_NAVY} !important; color: white !important;
    }}

    /* dataframe header */
    .dataframe thead th {{
        background: {MPA_NAVY} !important; color: white !important;
    }}
</style>
""", unsafe_allow_html=True)


# =========================================================================
# COUNTRY UNIVERSE
# =========================================================================
@dataclass
class Country:
    iso3: str
    iso2: str
    name: str
    group: str
    fred_ltir: Optional[str]

COUNTRIES: List[Country] = [
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
    Country("ISR", "IL", "Israel",         "AE", None),
    Country("MEX", "MX", "Mexico",         "EM", "IRLTLT01MXM156N"),
    Country("CHL", "CL", "Chile",          "EM", "IRLTLT01CLM156N"),
    Country("COL", "CO", "Colombia",       "EM", "IRLTLT01COM156N"),
    Country("PER", "PE", "Peru",           "EM", None),
    Country("BRA", "BR", "Brazil",         "EM", None),
    Country("ZAF", "ZA", "South Africa",   "EM", "IRLTLT01ZAM156N"),
    Country("TUR", "TR", "Turkey",         "EM", "IRLTLT01TRM156N"),
    Country("POL", "PL", "Poland",         "EM", "IRLTLT01PLM156N"),
    Country("CZE", "CZ", "Czech Republic", "EM", "IRLTLT01CZM156N"),
    Country("HUN", "HU", "Hungary",        "EM", "IRLTLT01HUM156N"),
    Country("ROU", "RO", "Romania",        "EM", None),
    Country("IND", "IN", "India",          "EM", "INDIRLTLT01STM"),
    Country("IDN", "ID", "Indonesia",      "EM", "IDNIRLTLT01STM"),
    Country("MYS", "MY", "Malaysia",       "EM", None),
    Country("PHL", "PH", "Philippines",    "EM", None),
    Country("THA", "TH", "Thailand",       "EM", None),
    Country("CHN", "CN", "China",          "EM", "INTGSTCNM193N"),
    Country("EGY", "EG", "Egypt",          "EM", None),
]

BENCHMARK_US = "USA"
BENCHMARK_DE = "DEU"

WB_INDICATORS = {
    "gdp_usd":     "NY.GDP.MKTP.CD",
    "gdp_growth":  "NY.GDP.MKTP.KD.ZG",
    "cpi_yoy":     "FP.CPI.TOTL.ZG",
    "debt_gdp":    "GC.DOD.TOTL.GD.ZS",
    "cab_gdp":     "BN.CAB.XOKA.GD.ZS",
    "reserves_mo": "FI.RES.TOTL.MO",
    "trade_gdp":   "NE.TRD.GNFS.ZS",
    "co2_kt":      "EN.ATM.CO2E.KT",
    "co2_pc":      "EN.ATM.CO2E.PC",
    "co2_gdp":     "EN.ATM.CO2E.KD.GD",
    "renew_share": "EG.FEC.RNEW.ZS",
    "energy_gdp":  "EG.USE.COMM.GD.PP.KD",
}

BIS_CREDIT_CSV = "https://data.bis.org/static/bulk/WS_TC_csv_flat.zip"


# =========================================================================
# HELPERS
# =========================================================================
def mpa_topbar(subtitle: str):
    st.markdown(f"""
    <div class="mpa-topbar">
        <div>
            <div class="title">THE MOUNTAIN PATH ACADEMY</div>
            <div class="sub">World of Finance &nbsp;|&nbsp; {subtitle}</div>
        </div>
        <div style="color: {MPA_YELLOW}; font-size: 0.85rem; font-weight: 600;">
            Sovereign Panel Builder &nbsp;•&nbsp; Research Tool
        </div>
    </div>
    """, unsafe_allow_html=True)


def mpa_box(kind: str, body_html: str):
    """kind in {'info','warn','ok','insight'}"""
    klass = {"info": "", "warn": "mpa-box-warn",
             "ok": "mpa-box-ok", "insight": "mpa-box-insight"}[kind]
    st.markdown(f'<div class="mpa-box {klass}">{body_html}</div>',
                unsafe_allow_html=True)


# =========================================================================
# DATA LOADERS (cached where it makes sense)
# =========================================================================
@st.cache_data(show_spinner=False, ttl=3600)
def load_fred_yields(api_key: str, iso3_to_ticker: Dict[str, str],
                     start: str, end: Optional[str]) -> pd.DataFrame:
    """Pull long-term yields from FRED.  Cached for 1 hour per (key, selection)."""
    from fredapi import Fred

    fred = Fred(api_key=api_key)
    frames = {}
    errors = {}
    for iso3, ticker in iso3_to_ticker.items():
        try:
            s = fred.get_series(ticker,
                                observation_start=start,
                                observation_end=end)
            frames[iso3] = s.rename(iso3)
        except Exception as e:
            errors[iso3] = str(e)
        time.sleep(0.12)

    if not frames:
        # signal upstream that every call failed (usually bad API key)
        raise RuntimeError(
            "No FRED series returned. The most common cause is an invalid "
            "API key. Example message from first failing series: "
            + (next(iter(errors.values())) if errors else "unknown")
        )

    df = pd.concat(frames.values(), axis=1).sort_index()
    df.index = pd.to_datetime(df.index)
    df.index.name = "date"
    return df, errors


@st.cache_data(show_spinner=False, ttl=3600)
def load_oecd_yields(iso2_list: tuple, start: str) -> pd.DataFrame:
    if not iso2_list:
        return pd.DataFrame()
    try:
        import pandasdmx as sdmx
    except ImportError:
        return pd.DataFrame()
    iso2_to_iso3 = {c.iso2: c.iso3 for c in COUNTRIES}
    oecd = sdmx.Request("OECD")
    frames = {}
    for iso2 in iso2_list:
        try:
            key = f"{iso2}.IRLT.ST.M"
            resp = oecd.data(resource_id="KEI", key=key,
                             params=dict(startTime=start[:7]))
            ser = resp.to_pandas()
            if ser.empty:
                continue
            if isinstance(ser.index, pd.MultiIndex):
                ser = ser.reset_index(
                    level=list(range(ser.index.nlevels - 1)), drop=True
                )
            ser.index = pd.to_datetime(ser.index)
            frames[iso2_to_iso3[iso2]] = ser.rename(iso2_to_iso3[iso2])
        except Exception:
            pass
        time.sleep(0.2)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames.values(), axis=1).sort_index()
    df.index.name = "date"
    return df


@st.cache_data(show_spinner=False, ttl=24 * 3600)
def load_worldbank_macro(iso3_list: tuple, start_year: int,
                         end_year: int) -> pd.DataFrame:
    try:
        import wbdata
    except ImportError:
        return pd.DataFrame()
    data_date = (datetime(start_year, 1, 1), datetime(end_year, 12, 31))
    try:
        try:
            df = wbdata.get_dataframe(
                indicators=WB_INDICATORS, country=list(iso3_list),
                date=data_date,
            )
        except TypeError:
            df = wbdata.get_dataframe(
                indicators=WB_INDICATORS, country=list(iso3_list),
                data_date=data_date,
            )
    except Exception as e:
        st.warning(f"World Bank fetch failed: {e}")
        return pd.DataFrame()

    df = df.reset_index().rename(columns={"country": "country_name", "date": "year"})
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    name_to_iso3 = {c.name: c.iso3 for c in COUNTRIES}
    df["iso3"] = df["country_name"].map(name_to_iso3)
    df = df.dropna(subset=["iso3", "year"]).drop(columns="country_name")
    col_order = ["iso3", "year"] + list(WB_INDICATORS.keys())
    return df[col_order].sort_values(["iso3", "year"]).reset_index(drop=True)


@st.cache_data(show_spinner=False, ttl=24 * 3600)
def load_bis_gov_debt(iso3_list: tuple) -> pd.DataFrame:
    try:
        import requests
    except ImportError:
        return pd.DataFrame()
    iso3_set = set(iso3_list)
    try:
        r = requests.get(BIS_CREDIT_CSV, timeout=90)
        r.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
            fname = [n for n in zf.namelist() if n.endswith(".csv")][0]
            with zf.open(fname) as f:
                df = pd.read_csv(f, low_memory=False)
    except Exception as e:
        st.warning(f"BIS download failed: {e}")
        return pd.DataFrame()
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
        return sub.dropna().sort_values(["iso3", "quarter"]).reset_index(drop=True)
    except Exception as e:
        st.warning(f"BIS filter failed: {e}")
        return pd.DataFrame()


# =========================================================================
# HARMONISATION
# =========================================================================
def monthly_to_quarterly(wide_monthly: pd.DataFrame) -> pd.DataFrame:
    if wide_monthly.empty:
        return pd.DataFrame(columns=["iso3", "quarter", "yield_10y"])
    try:
        q = wide_monthly.resample("QE").mean()
    except ValueError:
        q = wide_monthly.resample("Q").mean()
    q = q.rename_axis("date")
    q.index = q.index.to_period("Q")
    long = (q.stack().rename("yield_10y").reset_index()
             .rename(columns={"date": "quarter", "level_1": "iso3"}))
    return long[["iso3", "quarter", "yield_10y"]]


def build_spreads(yields_long: pd.DataFrame) -> pd.DataFrame:
    if yields_long.empty:
        return yields_long
    us = (yields_long.query("iso3 == @BENCHMARK_US")
          .set_index("quarter")["yield_10y"].rename("us_y"))
    de = (yields_long.query("iso3 == @BENCHMARK_DE")
          .set_index("quarter")["yield_10y"].rename("de_y"))
    out = yields_long.set_index("quarter").join([us, de])
    out["spread_vs_us_bp"] = (out["yield_10y"] - out["us_y"]) * 100
    out["spread_vs_de_bp"] = (out["yield_10y"] - out["de_y"]) * 100
    return (out.reset_index().drop(columns=["us_y", "de_y"])
            [["iso3", "quarter", "yield_10y",
              "spread_vs_us_bp", "spread_vs_de_bp"]])


def annual_to_quarterly(annual_long: pd.DataFrame,
                        id_cols=("iso3", "year")) -> pd.DataFrame:
    if annual_long.empty:
        return annual_long
    val_cols = [c for c in annual_long.columns if c not in id_cols]
    rows = []
    for _, r in annual_long.iterrows():
        y = int(r["year"])
        for q in range(1, 5):
            rec = {"iso3": r["iso3"],
                   "quarter": pd.Period(f"{y}Q{q}", freq="Q")}
            rec.update({c: r[c] for c in val_cols})
            rows.append(rec)
    return pd.DataFrame(rows)


# =========================================================================
# SIDEBAR — CONFIG
# =========================================================================
with st.sidebar:
    st.markdown(f"""
    <div style="text-align:center; padding: 15px 0 5px 0;">
        <div style="font-size: 1.15rem; font-weight: 800; color: white; letter-spacing: 1px;">
            THE MOUNTAIN PATH<br>ACADEMY
        </div>
        <div style="height: 2px; background: {MPA_YELLOW}; width: 60%; margin: 8px auto;"></div>
        <div style="color: {MPA_YELLOW}; font-size: 0.75rem; font-style: italic;">World of Finance</div>
    </div>
    """, unsafe_allow_html=True)

    st.header("Configuration")

    # -------- API key --------
    st.subheader("FRED API Key")
    st.markdown(
        '<div style="font-size: 0.8rem; color: #cfd8e3; margin-bottom: 6px;">'
        'Get a free key at '
        '<a href="https://fredaccount.stlouisfed.org/apikeys" '
        'target="_blank" style="color: #F4C430;">fredaccount.stlouisfed.org</a>'
        '</div>', unsafe_allow_html=True,
    )
    api_key = st.text_input(
        "Paste your FRED API key",
        type="password",
        placeholder="32-character hex string",
        help="Stored only in this browser session — never written to disk.",
        label_visibility="collapsed",
    )

    # -------- Date range --------
    st.subheader("Date range")
    start_year = st.number_input("Start year", min_value=1990,
                                  max_value=date.today().year,
                                  value=2005, step=1)
    end_year = st.number_input("End year", min_value=start_year,
                                max_value=date.today().year,
                                value=date.today().year, step=1)

    # -------- Country selection --------
    st.subheader("Countries")
    groups = st.multiselect(
        "Include groups",
        ["AE", "EM"], default=["AE", "EM"],
        help="AE = Advanced Economies, EM = Emerging Markets",
    )
    available = [c for c in COUNTRIES if c.group in groups]
    select_all = st.checkbox("Select all in groups", value=True)
    if select_all:
        selected_iso3 = [c.iso3 for c in available]
    else:
        selected_iso3 = st.multiselect(
            "Pick countries (ISO3)",
            [c.iso3 for c in available],
            default=[c.iso3 for c in available if c.fred_ltir],
        )

    # -------- Optional modules --------
    st.subheader("Data modules")
    incl_oecd = st.checkbox("OECD SDMX fallback (slow)", value=False,
                            help="Tries to fetch yields for countries not on FRED.")
    incl_wb   = st.checkbox("World Bank macro + CO₂", value=True)
    incl_bis  = st.checkbox("BIS government debt",    value=False,
                            help="Large download (~50 MB). Quarterly BIS total-credit stats.")

    st.markdown("---")
    st.caption("© Prof. V. Ravichandran · themountainpathacademy.com")


# =========================================================================
# MAIN PAGE
# =========================================================================
mpa_topbar("Climate Transition & Sovereign Debt — Data Builder")

st.markdown("""
<div style="text-align: center; padding: 20px 0 10px 0;">
    <div style="font-size: 2.3rem; font-weight: 800; color: {nav}; line-height: 1.1;">
        Sovereign Panel Builder
    </div>
    <div style="font-size: 1.0rem; color: {grey}; max-width: 820px; margin: 12px auto;">
        Assemble a country-quarter panel of long-term sovereign yields, spreads,
        macro controls and climate indicators — directly from FRED, OECD, World
        Bank and BIS. Ready to merge with your climate-transition variables.
    </div>
    <div style="height: 2px; background: {yel}; width: 200px; margin: 12px auto;"></div>
</div>
""".format(nav=MPA_NAVY, grey=MPA_GREY, yel=MPA_YELLOW), unsafe_allow_html=True)

# ---- Pre-flight checks ----
col1, col2, col3, col4 = st.columns(4)
col1.metric("Countries selected", len(selected_iso3))
col2.metric("Date range", f"{start_year} – {end_year}")
col3.metric("FRED tickers",
            sum(1 for c in COUNTRIES
                if c.iso3 in selected_iso3 and c.fred_ltir))
col4.metric("API key", "✔ set" if api_key else "— missing")

# ---- Instruction box ----
if not api_key:
    mpa_box("warn",
        "<b>Step 1 —</b> Paste your <b>FRED API key</b> in the sidebar. "
        "Get a free one in two minutes at "
        "<a href='https://fredaccount.stlouisfed.org/apikeys' target='_blank'>"
        "fredaccount.stlouisfed.org/apikeys</a>. "
        "The key stays in your browser session and is never written to disk.")
elif not selected_iso3:
    mpa_box("warn", "Select at least one country in the sidebar to build the panel.")
else:
    mpa_box("ok",
        f"Ready to build. Will pull yields for <b>{len(selected_iso3)}</b> "
        "countries, macro controls from the World Bank"
        + (", and BIS debt stats" if incl_bis else "")
        + ". Click <b>Build Panel</b> below.")

# ---- Build button ----
st.markdown("<div style='margin: 20px 0;'></div>", unsafe_allow_html=True)
ccol1, ccol2, ccol3 = st.columns([1, 1, 1])
with ccol2:
    build_clicked = st.button("🚀  Build Panel", use_container_width=True,
                              disabled=(not api_key or not selected_iso3))

# =========================================================================
# BUILD PIPELINE
# =========================================================================
if build_clicked:
    start_str = f"{start_year}-01-01"
    end_str   = f"{end_year}-12-31"

    iso3_to_ticker = {c.iso3: c.fred_ltir for c in COUNTRIES
                      if c.iso3 in selected_iso3 and c.fred_ltir}
    iso2_missing = tuple(
        c.iso2 for c in COUNTRIES
        if c.iso3 in selected_iso3 and not c.fred_ltir
    )
    selected_tuple = tuple(sorted(selected_iso3))

    status = st.status("Building panel …", expanded=True)

    # ---- FRED ----
    with status:
        st.write("**Step 1 / 5** — Pulling FRED long-term bond yields …")
        try:
            fred_wide, fred_errors = load_fred_yields(
                api_key, iso3_to_ticker, start_str, end_str
            )
            st.write(f"✔ {fred_wide.shape[1]} countries  ·  "
                     f"{fred_wide.shape[0]} monthly rows")
            if fred_errors:
                st.write(f"⚠ {len(fred_errors)} series failed: "
                         + ", ".join(fred_errors.keys()))
        except Exception as e:
            status.update(label="FRED fetch failed", state="error")
            st.error(f"FRED fetch failed:\n\n{e}")
            st.stop()

    # ---- OECD fallback ----
    with status:
        if incl_oecd and iso2_missing:
            st.write(f"**Step 2 / 5** — OECD SDMX fallback for "
                     f"{len(iso2_missing)} countries …")
            oecd_wide = load_oecd_yields(iso2_missing, start_str)
            if not oecd_wide.empty:
                st.write(f"✔ {oecd_wide.shape[1]} extra series from OECD")
            else:
                st.write("⚠ OECD returned nothing (API may be down or countries not covered)")
        else:
            st.write("**Step 2 / 5** — OECD fallback skipped")
            oecd_wide = pd.DataFrame()

    yields_wide = (fred_wide.combine_first(oecd_wide)
                   if not oecd_wide.empty else fred_wide)

    # ---- Quarterly aggregation + spreads ----
    with status:
        st.write("**Step 3 / 5** — Aggregating to quarterly & computing spreads …")
        yields_q = monthly_to_quarterly(yields_wide)
        yields_q = build_spreads(yields_q)
        st.write(f"✔ {len(yields_q):,} country-quarter yield rows  ·  "
                 f"{yields_q['quarter'].min()} → {yields_q['quarter'].max()}")

    # ---- World Bank ----
    with status:
        if incl_wb:
            st.write("**Step 4 / 5** — World Bank macro + CO₂ indicators …")
            wb_annual = load_worldbank_macro(
                selected_tuple, int(start_year), int(end_year)
            )
            wb_q = annual_to_quarterly(wb_annual)
            st.write(f"✔ {len(wb_q):,} country-quarter macro rows")
        else:
            st.write("**Step 4 / 5** — World Bank skipped")
            wb_q = pd.DataFrame()

    # ---- BIS ----
    with status:
        if incl_bis:
            st.write("**Step 5 / 5** — BIS government debt (large download, ~50 MB) …")
            bis_q = load_bis_gov_debt(selected_tuple)
            st.write(f"✔ {len(bis_q):,} BIS rows for selected countries")
        else:
            st.write("**Step 5 / 5** — BIS skipped")
            bis_q = pd.DataFrame()

    # ---- Merge ----
    panel = yields_q.copy()
    if not wb_q.empty:
        panel = panel.merge(wb_q, on=["iso3", "quarter"], how="left")
    if not bis_q.empty:
        panel = panel.merge(bis_q, on=["iso3", "quarter"], how="left")

    meta = pd.DataFrame(
        [(c.iso3, c.iso2, c.name, c.group) for c in COUNTRIES],
        columns=["iso3", "iso2", "country_name", "group"],
    )
    panel = panel.merge(meta, on="iso3", how="left")
    lead = ["iso3", "iso2", "country_name", "group", "quarter",
            "yield_10y", "spread_vs_us_bp", "spread_vs_de_bp"]
    other = [c for c in panel.columns if c not in lead]
    panel = (panel[lead + other]
             .sort_values(["iso3", "quarter"]).reset_index(drop=True))

    status.update(label=f"✅ Panel built — {len(panel):,} rows × {panel.shape[1]} cols",
                  state="complete")
    st.session_state["panel"] = panel


# =========================================================================
# RESULTS
# =========================================================================
if "panel" in st.session_state:
    panel: pd.DataFrame = st.session_state["panel"]

    st.markdown('<div class="mpa-section">Panel summary</div>',
                unsafe_allow_html=True)

    m = st.columns(5)
    m[0].metric("Countries",  panel["iso3"].nunique())
    m[1].metric("Quarters",   panel["quarter"].nunique())
    m[2].metric("Rows",       f"{len(panel):,}")
    m[3].metric("Columns",    panel.shape[1])
    yld_cov = panel["yield_10y"].notna().mean() * 100
    m[4].metric("Yield coverage", f"{yld_cov:.1f}%")

    tab1, tab2, tab3, tab4 = st.tabs(
        ["📊 Preview", "📈 Visualise", "🗺 Coverage", "⬇ Download"]
    )

    # ---- Preview ----
    with tab1:
        st.markdown("**First 200 rows**")
        preview = panel.head(200).copy()
        preview["quarter"] = preview["quarter"].astype(str)
        st.dataframe(preview, use_container_width=True, hide_index=True)

        st.markdown("**Column dictionary**")
        dic = pd.DataFrame([
            ("iso3 / iso2",           "Country codes"),
            ("country_name / group",  "Country name and AE/EM tag"),
            ("quarter",               "Calendar quarter (YYYYQn)"),
            ("yield_10y",             "10-year benchmark sovereign yield, %"),
            ("spread_vs_us_bp",       "Yield − US 10Y Treasury, basis points"),
            ("spread_vs_de_bp",       "Yield − German 10Y Bund, basis points"),
            ("gdp_usd",               "GDP (current US$)"),
            ("gdp_growth",            "Real GDP growth, % YoY"),
            ("cpi_yoy",               "Consumer price inflation, % YoY"),
            ("debt_gdp",              "Central government debt, % of GDP"),
            ("cab_gdp",               "Current account balance, % of GDP"),
            ("reserves_mo",           "Reserves in months of imports"),
            ("trade_gdp",             "Trade openness, % of GDP"),
            ("co2_kt",                "Total CO₂ emissions, kilotons"),
            ("co2_pc",                "CO₂ per capita, tons"),
            ("co2_gdp",               "kg CO₂ per 2015 USD of GDP (carbon intensity)"),
            ("renew_share",           "Renewables, % of final energy consumption"),
            ("energy_gdp",            "Energy intensity (kg oil-eq per 2017 PPP $)"),
            ("bis_gov_debt_pct_gdp",  "BIS general-gov debt at market value, % GDP"),
        ], columns=["column", "description"])
        st.dataframe(dic, use_container_width=True, hide_index=True)

    # ---- Visualise ----
    with tab2:
        vp = panel.copy()
        vp["quarter"] = vp["quarter"].dt.to_timestamp()

        st.markdown("**10-year yield over time**")
        default_sel = sorted(vp["iso3"].unique())[:8]
        chart_iso = st.multiselect(
            "Countries to plot", sorted(vp["iso3"].unique()),
            default=default_sel,
        )
        if chart_iso:
            sub = vp[vp["iso3"].isin(chart_iso)]
            fig = px.line(sub, x="quarter", y="yield_10y", color="iso3",
                          labels={"yield_10y": "10Y yield (%)", "quarter": ""},
                          template="simple_white")
            fig.update_layout(height=450,
                              title="Sovereign 10-year yields",
                              title_font=dict(color=MPA_NAVY, size=16),
                              legend_title_text="Country")
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("**Spread vs US Treasury (basis points)**")
            fig2 = px.line(sub, x="quarter", y="spread_vs_us_bp", color="iso3",
                           labels={"spread_vs_us_bp": "Spread (bp)", "quarter": ""},
                           template="simple_white")
            fig2.update_layout(height=400,
                               title="Spread vs. US 10Y Treasury",
                               title_font=dict(color=MPA_NAVY, size=16),
                               legend_title_text="Country")
            fig2.add_hline(y=0, line=dict(color=MPA_GREY, dash="dash"))
            st.plotly_chart(fig2, use_container_width=True)

    # ---- Coverage heatmap ----
    with tab3:
        st.markdown("**Data coverage heatmap — % non-null per variable / country**")

        cov_cols = [c for c in panel.columns
                    if c not in ("iso3", "iso2", "country_name", "group", "quarter")]

        # Flag missing blocks so the user knows *why* the heatmap is narrow.
        yield_cols  = [c for c in cov_cols if c in ("yield_10y", "spread_vs_us_bp", "spread_vs_de_bp")]
        macro_cols  = [c for c in cov_cols if c in ("gdp_usd", "gdp_growth", "cpi_yoy",
                                                     "debt_gdp", "cab_gdp", "reserves_mo", "trade_gdp")]
        clim_cols   = [c for c in cov_cols if c in ("co2_kt", "co2_pc", "co2_gdp",
                                                     "renew_share", "energy_gdp")]
        bis_cols    = [c for c in cov_cols if c.startswith("bis_")]

        missing_blocks = []
        if not macro_cols: missing_blocks.append("macro (World Bank WDI)")
        if not clim_cols:  missing_blocks.append("climate (World Bank CO₂ / energy)")
        if not bis_cols:   missing_blocks.append("BIS government debt")
        if missing_blocks:
            mpa_box(
                "warn",
                "<b>Only the yield block is populated.</b> These modules returned no data: "
                + ", ".join(f"<code>{m}</code>" for m in missing_blocks)
                + ".<br>Check the sidebar toggles are on, and that the build log above didn't show red "
                "warnings for World Bank / BIS fetches. World Bank and BIS don't need an API key — "
                "they can fail intermittently if the host is slow."
            )

        cov_mat = (panel.groupby("iso3")[cov_cols]
                        .apply(lambda d: d.notna().mean() * 100)
                        .round(1))

        fig = go.Figure(data=go.Heatmap(
            z=cov_mat.values,
            x=cov_mat.columns,
            y=cov_mat.index,
            colorscale=[[0, "#FBE5E7"], [0.5, "#FFF6DD"], [1, "#E3F1E8"]],
            zmin=0, zmax=100,
            colorbar=dict(title="%"),
            hovertemplate="<b>%{y}</b><br>%{x}: %{z:.1f}%<extra></extra>",
        ))
        fig.update_layout(
            title=dict(
                text=f"Coverage across {len(cov_mat)} countries × {len(cov_cols)} variables",
                font=dict(color=MPA_NAVY, size=15),
                x=0.0, xanchor="left",
            ),
            height=max(400, 22 * len(cov_mat)),
            margin=dict(l=60, r=20, t=60, b=60),
            xaxis=dict(side="bottom", tickangle=-30),
            yaxis=dict(autorange="reversed"),
        )
        st.plotly_chart(fig, use_container_width=True)

    # ---- Download ----
    with tab4:
        panel_out = panel.copy()
        panel_out["quarter"] = panel_out["quarter"].astype(str)
        csv_wide = panel_out.to_csv(index=False).encode("utf-8")

        long_out = panel_out.melt(
            id_vars=["iso3", "iso2", "country_name", "group", "quarter"],
            var_name="variable", value_name="value",
        ).dropna(subset=["value"])
        csv_long = long_out.to_csv(index=False).encode("utf-8")

        try:
            import pyarrow  # noqa
            buf = io.BytesIO()
            panel_out.to_parquet(buf, index=False)
            parq_bytes = buf.getvalue()
            parq_ok = True
        except Exception:
            parq_bytes = b""
            parq_ok = False

        mpa_box("insight",
            "<b>Three ready-to-use formats:</b><br>"
            "• <b>Wide CSV</b> — one row per country-quarter, ideal for Excel & panel regressions.<br>"
            "• <b>Long CSV</b> — tidy format (country, quarter, variable, value), ideal for ggplot / seaborn.<br>"
            "• <b>Parquet</b> — compressed columnar format for pandas / polars."
        )

        d1, d2, d3 = st.columns(3)
        d1.download_button(
            "⬇ Wide CSV", data=csv_wide,
            file_name="sovereign_panel_wide.csv",
            mime="text/csv", use_container_width=True,
        )
        d2.download_button(
            "⬇ Long CSV", data=csv_long,
            file_name="sovereign_panel_long.csv",
            mime="text/csv", use_container_width=True,
        )
        if parq_ok:
            d3.download_button(
                "⬇ Parquet", data=parq_bytes,
                file_name="sovereign_panel.parquet",
                mime="application/octet-stream", use_container_width=True,
            )
        else:
            d3.button("Parquet unavailable (pip install pyarrow)",
                      disabled=True, use_container_width=True)

        st.markdown("---")
        mpa_box("insight",
            "<b>Next step for your dissertation:</b> merge your climate-transition "
            "variable (NGFS scenario delta, OECD EPS index, ND-GAIN score, etc.) on "
            "<code>(iso3, quarter)</code> and run two-way fixed-effects or panel "
            "local projections with <code>yield_10y</code> or "
            "<code>spread_vs_us_bp</code> on the LHS.")

# =========================================================================
# FOOTER
# =========================================================================
st.markdown(f"""
<div style="border-top: 2px solid {MPA_YELLOW}; margin-top: 2rem;
            padding-top: 10px; color: {MPA_GREY}; font-size: 0.85rem;
            display: flex; justify-content: space-between;">
    <span>&copy; 2026 Prof. V. Ravichandran · The Mountain Path Academy</span>
    <span><em>themountainpathacademy.com</em></span>
</div>
""", unsafe_allow_html=True)

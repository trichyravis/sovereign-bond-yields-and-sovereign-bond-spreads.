# Sovereign Bond Yield & Spread Panel Builder

A starter Python pipeline for assembling a country-quarter panel of sovereign
long-term bond yields, spreads (vs. US Treasury and German Bund), macro
controls, and selected climate indicators for **40 countries** (23 AE + 17 EM).

Designed for dissertation research on **climate transition and sovereign debt**.

---

## 1. One-time setup

```bash
# clone / copy the two files into your project folder:
#   sovereign_panel_builder.py
#   requirements-panel.txt

python -m venv venv
source venv/bin/activate           # Windows:  venv\Scripts\activate
pip install -r requirements-panel.txt
```

### Get a free FRED API key
1. Register at <https://fredaccount.stlouisfed.org/apikeys>  (takes 2 minutes).
2. Export it before running:
   ```bash
   export FRED_API_KEY="xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"      # mac/linux
   setx FRED_API_KEY "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"        # windows
   ```

No key is needed for World Bank, OECD, or BIS.

---

## 2. Run

```bash
python sovereign_panel_builder.py
```

The script prints a per-country progress log and writes three files to
`./outputs_panel/`:

| File | Description |
|---|---|
| `sovereign_panel_wide.csv`     | country-quarter panel, one row per country-quarter |
| `sovereign_panel_long.csv`     | tidy long format (country, quarter, variable, value) |
| `sovereign_panel.parquet`      | same wide frame, compressed, fast to re-load in pandas |

---

## 3. What's in the panel

### Identifiers
`iso3`, `iso2`, `country_name`, `group` (AE/EM), `quarter`

### Yield block (primary FRED, fallback OECD SDMX)
* `yield_10y` — 10-year government benchmark, % (quarter-average)
* `spread_vs_us_bp` — yield − US 10Y, basis points
* `spread_vs_de_bp` — yield − German 10Y (Bund), basis points

### Macro block (World Bank WDI, annual → forward-filled to quarterly)
* `gdp_usd`, `gdp_growth`, `cpi_yoy`
* `debt_gdp`  (central government debt, % GDP)
* `cab_gdp`   (current account, % GDP)
* `reserves_mo`, `trade_gdp`

### Climate block (World Bank WDI)
* `co2_kt`     — total CO₂ emissions, kt
* `co2_pc`     — per capita
* `co2_gdp`    — kg CO₂ per 2015 USD of GDP (carbon intensity)
* `renew_share` — renewables as % of final energy
* `energy_gdp` — energy intensity

### Debt block (BIS)
* `bis_gov_debt_pct_gdp` — general government, at market value, quarterly

---

## 4. Countries covered

**Advanced Economies (23):** USA, GBR, DEU, FRA, ITA, ESP, NLD, BEL, AUT, IRL,
PRT, GRC, FIN, SWE, DNK, NOR, CHE, CAN, AUS, NZL, JPN, KOR, ISR.

**Emerging Markets (17):** MEX, CHL, COL, PER, BRA, ZAF, TUR, POL, CZE, HUN,
ROU, IND, IDN, MYS, PHL, THA, CHN, EGY.

Countries marked `fred_ltir = None` in `COUNTRIES` will try the OECD SDMX
fallback. A handful (BRA, MYS, PHL, THA, EGY) are only reliably available via
Bloomberg/Refinitiv — the script will report these as missing so you can
backfill them from your university's subscription.

---

## 5. Merging your climate-transition variables

Whatever climate-transition dataset you have (NGFS scenarios, MCC policy index,
Climate Policy Radar, Verisk Maplecroft, ND-GAIN, etc.), format it to
`(iso3, quarter)` and merge:

```python
import pandas as pd
panel   = pd.read_parquet("outputs_panel/sovereign_panel.parquet")
panel["quarter"] = pd.PeriodIndex(panel["quarter"], freq="Q")
climate = pd.read_csv("your_climate_data.csv")        # must have iso3, quarter
climate["quarter"] = pd.PeriodIndex(climate["quarter"], freq="Q")

full = panel.merge(climate, on=["iso3", "quarter"], how="left")
```

---

## 6. Extending the panel

Common extensions for this literature:

* **Sovereign CDS spreads (5Y)** — add a loader for Refinitiv/Bloomberg, schema `(iso3, quarter, cds_5y_bp)`.
* **JPM EMBI Global Diversified** — Bloomberg (`JPEGCOMP Index`) or Refinitiv.
* **Rating scores** — S&P / Moody's / Fitch numeric scales.
* **Climate policy stringency** — OECD EPS index, ClimateLaws.org database.
* **Physical & transition risk scores** — NGFS scenarios, Climate Bonds Initiative.

Each is a separate loader → merge on `(iso3, quarter)`.

---

## 7. Known limitations

* FRED's OECD LTIR series are published with ~1-month lag.
* World Bank WDI is annual; quarterly interpolation assumes within-year constancy — document this in your methodology.
* BIS flat files are occasionally restructured; if the BIS block returns empty, check `BIS_CREDIT_CSV` URL at <https://data.bis.org/>.
* OECD SDMX migrated to a new endpoint in 2024; `pandasdmx` covers it but some country keys changed. Adjust the `key` string in `load_oecd_yields` if a country fails.

---

© 2026 Prof. V. Ravichandran · The Mountain Path Academy · World of Finance

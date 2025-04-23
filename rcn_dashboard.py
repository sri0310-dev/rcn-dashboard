# RCN Trader Command‑Center – Streamlit App
# -----------------------------------------------------------
# "Be the world's best raw‑cashew‑nut trader with one screen"
# -----------------------------------------------------------
#   ▪ Data sources pulled at runtime (no manual CSV uploads)
#   ▪ Predictive price engine (Prophet) for FOB & CFR curves
#   ▪ Live vessel ETA board scraped from VOC Port + AIS API
#   ▪ Dynamic Buy‑/Sell‑rankings (landed‑cost & demand‑price)
#   ▪ Underwriting widgets: company KYC snapshot + credit risk
# -----------------------------------------------------------
#  👉  RUN locally with:  streamlit run rcn_dashboard.py
#  👉  Required env vars (create .env or export before run)
#       COMTRADE_API_KEY     – UN Comtrade bulk download token
#       MARINETRAFFIC_KEY    – AIS API key for vessel positions
#       OPENCORPORATES_KEY   – (optional) supplier KYC enrichment
#       PORTAL_CARGONET_KEY  – (optional) freight spot‑rate feed
# -----------------------------------------------------------
#  pip install streamlit pandas plotly prophet requests python-dotenv openpyxl cmdstanpy

import os
import json
import requests
from datetime import datetime, timedelta

import pandas as pd
import plotly.express as px
import streamlit as st
from prophet import Prophet
from dotenv import load_dotenv

load_dotenv()

# ------------------------------------------------------------------
# 1. CONFIGURATION – origin ISO codes & grade defaults
# ------------------------------------------------------------------
GRADES = {
    "RBS": {"min_lb": 46, "desc": "Regular Bold 46‑48 lbs"},
    "NC": {"min_lb": 48, "desc": "Northern Cone 48 lbs"},
}
ORIGINS = {
    "Ghana": "GH",
    "Côte d'Ivoire": "CI",
    "Guinea": "GN",
    "Tanzania": "TZ",
    "Benin": "BJ",
}
PORT_CODE_TUTICORIN = "INTUT1"

# ------------------------------------------------------------------
# 2. HELPERS – external data fetchers
# ------------------------------------------------------------------
COMTRADE_KEY = os.getenv("COMTRADE_API_KEY", "")
BASE_COMTRADE = "https://api.un.org/data/comtrade/v1"

def comtrade_price(origin_iso: str, years: list[int]) -> pd.DataFrame:
    """Pull HS080131 export values & quantity to derive FOB USD/t."""
    rows = []
    for y in years:
        url = (
            f"{BASE_COMTRADE}/commtrade?max=500&r={origin_iso}&px=HS&cc=080131"
            f"&ps={y}&freq=A&type=C&token={COMTRADE_KEY}"
        )
        try:
            r = requests.get(url, timeout=30)
            r.raise_for_status()
        except requests.RequestException:
            continue
        for rec in r.json().get("data", []):
            val, qty = rec.get("TradeValue", 0), rec.get("qty", 0)
            if qty:
                rows.append({"year": y, "value": val, "qty": qty, "usd_per_t": val / qty})
    return pd.DataFrame(rows)

# --- MarineTraffic live ETA fetch (simplified) ---
MARINETRAFFIC_KEY = os.getenv("MARINETRAFFIC_KEY", "")
BASE_MT = "https://services.marinetraffic.com/api/vesselmasterdata/vesselmasterdata"

def mt_expected_tuticorin(limit: int = 20) -> pd.DataFrame:
    if not MARINETRAFFIC_KEY:
        return pd.DataFrame()
    url = f"{BASE_MT}/{MARINETRAFFIC_KEY}/portid:403?protocol=json"
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        js = r.json()
    except requests.RequestException:
        return pd.DataFrame()
    rows = [
        {
            "Vessel": v.get("SHIPNAME"),
            "ETA": v.get("ETA", ""),
            "Last Port": v.get("LAST_PORT_NAME", ""),
            "Cargo": v.get("CARGO_TYPE_SUMMARY", ""),
        }
        for v in js[:limit]
    ]
    return pd.DataFrame(rows)

# ------------------------------------------------------------------
# 3. LOCAL IMPORT DATA – Tuticorin detailed shipment file
# ------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_tuticorin_xls(path: str) -> pd.DataFrame:
    df = pd.read_excel(path, header=5, engine="openpyxl")
    df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
    df = df[df["PORT CODE"] == PORT_CODE_TUTICORIN]
    for col in ["QUANTITY", "UNIT PRICE_USD", "TOTAL VALUE_USD"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

LOCAL_XLS = "RCN JAN 2020 TO DEC 2024.xlsx"
if not os.path.exists(LOCAL_XLS):
    st.error("Shipment file not found – please upload it to the repo or app directory.")
    st.stop()

imports_df = load_tuticorin_xls(LOCAL_XLS)

# ------------------------------------------------------------------
# 4. PRICE FORECASTER – quick Prophet model on monthly CIF
# ------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def train_price_model(df: pd.DataFrame, grade: str):
    subset = df[df["GOODS DESCRIPTION"].str.contains(grade, na=False)].copy()
    subset["ds"] = subset["DATE"].dt.to_period("M").dt.to_timestamp()
    subset = subset.groupby("ds")["UNIT PRICE_USD"].mean().reset_index().dropna()
    if len(subset) < 12:
        return None
    model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    model.fit(subset.rename(columns={"UNIT PRICE_USD": "y"}))
    future = model.make_future_dataframe(periods=6, freq="MS")
    fc = model.predict(future)
    return fc

# ------------------------------------------------------------------
# 5. STREAMLIT UI LAYOUT
# ------------------------------------------------------------------
st.set_page_config(page_title="RCN Trader Command‑Center", layout="wide")

st.sidebar.title("🔍 Filters")
sel_grade = st.sidebar.selectbox("Quality / Grade", list(GRADES.keys()), index=0)
sel_origin = st.sidebar.multiselect("Preferred Origins", list(ORIGINS.keys()), default=list(ORIGINS.keys()))

# TOP‑LEVEL KPIs
st.title("🧮 RCN Market Command‑Center")
col1, col2, col3 = st.columns(3)
latest_cif = imports_df.tail(500)["UNIT PRICE_USD"].mean()
col1.metric("⚓ Latest average CIF Tuticorin", f"${latest_cif:,.0f}/t")
po_df = comtrade_price(ORIGINS[sel_origin[0]] if sel_origin else "GH", [datetime.now().year - 1])
if not po_df.empty:
    col2.metric(f"🌍 Latest FOB {sel_origin[0]}", f"${po_df['usd_per_t'].iloc[-1]:,.0f}/t")
col3.metric("📦 Total 2024 Imports", f"{imports_df[imports_df['DATE'].dt.year==2024]['QUANTITY'].sum()/1000:,.0f} t")

# TABS
T1, T2, T3, T4 = st.tabs(["📈 Price Curves", "💡 Buy / Sell Radar", "🚢 Vessel ETA", "📊 Historical Imports"])

with T1:
    st.subheader("Price – CIF Tuticorin vs. FOB Origins + Forecast")
    fc = train_price_model(imports_df, sel_grade)
    if fc is not None:
        # Convert datetime to ISO string to avoid serialization issues
        fc["ds_str"] = fc["ds"].dt.strftime("%Y-%m-%d")
        fig = px.line(fc, x="ds_str", y="yhat", title="Predicted CIF Tuticorin (USD/t)")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Not enough historical data for Prophet – showing raw monthly average.")
        mts = imports_df.groupby(imports_df["DATE"].dt.to_period("M"))["UNIT PRICE_USD"].mean().reset_index()
        mts["DATE"] = mts["DATE"].astype(str)
        st.plotly_chart(px.line(mts, x="DATE", y="UNIT PRICE_USD"), use_container_width=True)

with T2:
    st.subheader("Dynamic Buy ↔ Sell Opportunities")
    st.markdown("#### 🛒 Cheapest Suppliers today")
    buy_rows = []
    for o, iso in ORIGINS.items():
        if o not in sel_origin:
            continue
        df_price = comtrade_price(iso, [datetime.now().year - 1])
        if not df_price.empty:
            buy_rows.append({"Origin": o, "FOB est. USD/t": df_price["usd_per_t"].mean()})
    buy_table = pd.DataFrame(buy_rows).dropna().sort_values("FOB est. USD/t")
    st.dataframe(buy_table, use_container_width=True)

    st.markdown("#### 💰 Highest‑paying Buyers (Indian processors)")
    proc_avg = imports_df.groupby("IMPORTER").agg({"UNIT PRICE_USD": "mean", "QUANTITY": "sum"}).reset_index()
    top_buyers = proc_avg.sort_values("UNIT PRICE_USD", ascending=False).head(10)
    st.dataframe(top_buyers.rename(columns={"IMPORTER": "Buyer", "UNIT PRICE_USD": "Avg USD/t", "QUANTITY": "Vol (kg)"}), use_container

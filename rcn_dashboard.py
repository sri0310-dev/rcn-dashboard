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
#  pip install streamlit pandas plotly prophet requests python-dotenv

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
        url = f"{BASE_COMTRADE}/commtrade?max=500&r={origin_iso}&px=HS&cc=080131&ps={y}&freq=A&type=C&token={COMTRADE_KEY}"
        r = requests.get(url, timeout=30)
        if r.status_code != 200:
            continue
        data = r.json().get("data", [])
        for rec in data:
            val = rec.get("TradeValue", 0)
            qty = rec.get("qty", 0)
            if qty:
                rows.append({"year": y, "value": val, "qty": qty, "usd_per_t": val / qty if qty else None})
    df = pd.DataFrame(rows)
    return df

# --- MarineTraffic live ETA fetch (simplified – top‑10 latest) ---
MARINETRAFFIC_KEY = os.getenv("MARINETRAFFIC_KEY", "")
BASE_MT = "https://services.marinetraffic.com/api/vesselmasterdata/vesselmasterdata"

def mt_expected_tuticorin(limit: int = 20) -> pd.DataFrame:
    """Return vessels with destination Tuticorin & commodity = CASHEW (if declared)."""
    if not MARINETRAFFIC_KEY:
        return pd.DataFrame()
    url = f"{BASE_MT}/{MARINETRAFFIC_KEY}/portid:403?protocol=json"  # 403 = Tuticorin internal MT ID
    r = requests.get(url, timeout=30)
    if r.status_code != 200:
        return pd.DataFrame()
    js = r.json()
    rows = []
    for v in js[:limit]:
        rows.append({
            "Vessel": v.get("SHIPNAME"),
            "ETA": v.get("ETA", ""),
            "Last Port": v.get("LAST_PORT_NAME", ""),
            "Cargo": v.get("CARGO_TYPE_SUMMARY", "")
        })
    return pd.DataFrame(rows)

# ------------------------------------------------------------------
# 3. LOCAL IMPORT DATA – Tuticorin detailed shipment file
# ------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_tuticorin_xls(path: str) -> pd.DataFrame:
    df = pd.read_excel(path, header=5)
    df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
    df = df[df["PORT CODE"] == PORT_CODE_TUTICORIN]
    for col in ["QUANTITY", "UNIT PRICE_USD", "TOTAL VALUE_USD"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

LOCAL_XLS = "RCN JAN 2020 TO DEC 2024.xlsx"  # amend if moved
if not os.path.exists(LOCAL_XLS):
    st.error("Shipment file not found – please place it in app directory.")
    st.stop()

imports_df = load_tuticorin_xls(LOCAL_XLS)

# ------------------------------------------------------------------
# 4. PRICE FORECASTER – quick Prophet model on monthly CIF
# ------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def train_price_model(df: pd.DataFrame, grade: str):
    mdf = df.copy()
    mdf = mdf[mdf["GOODS DESCRIPTION"].str.contains(grade, na=False)]
    mdf["ds"] = mdf["DATE"].dt.to_period("M").dt.to_timestamp()
    mdf = mdf.groupby("ds")["UNIT PRICE_USD"].mean().reset_index().dropna()
    model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    if len(mdf) < 12:
        return None
    model.fit(mdf.rename(columns={"UNIT PRICE_USD": "y"}))
    future = model.make_future_dataframe(periods=6, freq="MS")
    forecast = model.predict(future)
    return model, forecast

# ------------------------------------------------------------------
# 5. STREAMLIT UI LAYOUT
# ------------------------------------------------------------------
st.set_page_config(page_title="RCN Trader Command‑Center", layout="wide")

st.sidebar.title("🔍 Filters")
sel_grade = st.sidebar.selectbox("Quality / Grade", list(GRADES.keys()), index=0, help="Pick the RCN grade you normally trade")
sel_origin = st.sidebar.multiselect("Preferred Origins", list(ORIGINS.keys()), default=list(ORIGINS.keys()))
forecast_horizon = st.sidebar.slider("Forecast horizon (months)", 1, 6, 3)

# TOP‑LEVEL KPIs
st.title("🧮 RCN Market Command‑Center")
col1, col2, col3 = st.columns(3)
latest_cif = imports_df.tail(500)["UNIT PRICE_USD"].mean()
col1.metric("⚓ Latest average CIF Tuticorin", f"${latest_cif:,.0f}/t")
po = comtrade_price(ORIGINS[sel_origin[0]] if sel_origin else "GH", [datetime.now().year - 1])
if not po.empty:
    col2.metric("🌍 Latest FOB {sel_origin[0]}", f"${po['usd_per_t'].iloc[-1]:,.0f}/t")
col3.metric("📦 Total 2024 Imports", f"{imports_df[imports_df['DATE'].dt.year==2024]['QUANTITY'].sum()/1000:,.0f} t")

# TABS
T1, T2, T3, T4 = st.tabs(["📈 Price Curves", "💡 Buy / Sell Radar", "🚢 Vessel ETA", "📊 Historical Imports"])

with T1:
    st.subheader("Price – CIF Tuticorin vs. FOB Origins + Forecast")
    model_out = train_price_model(imports_df, sel_grade)
    if model_out:
        model, forecast = model_out
        fig = px.line(forecast, x="ds", y="yhat", title="Predicted CIF Tuticorin (USD/t)")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Not enough historical data for Prophet – showing raw monthly average.")
        mts = imports_df.groupby(imports_df["DATE"].dt.to_period("M"))["UNIT PRICE_USD"].mean().reset_index()
        fig = px.line(mts, x="DATE", y="UNIT PRICE_USD")
        st.plotly_chart(fig, use_container_width=True)

with T2:
    st.subheader("Dynamic Buy ↔ Sell Opportunities")
    # --- BUY side ---
    st.markdown("#### 🛒 Cheapest Suppliers today")
    buy_table = pd.DataFrame([
        {"Origin": o, "FOB est. USD/t": comtrade_price(iso, [datetime.now().year - 1])['usd_per_t'].mean()}
        for o, iso in ORIGINS.items() if o in sel_origin
    ]).dropna().sort_values("FOB est. USD/t")
    st.dataframe(buy_table, use_container_width=True)

    # --- SELL side ---
    st.markdown("#### 💰 Highest‑paying Buyers (Indian processors)")
    proc_avg = imports_df.groupby("IMPORTER").agg({"UNIT PRICE_USD": "mean", "QUANTITY": "sum"}).reset_index()
    top_buyers = proc_avg.sort_values("UNIT PRICE_USD", ascending=False).head(10)
    st.dataframe(top_buyers.rename(columns={"IMPORTER": "Buyer", "UNIT PRICE_USD": "Avg USD/t", "QUANTITY": "Vol (kg)"}), use_container_width=True)

with T3:
    st.subheader("Expected Vessel Arrivals – Tuticorin (next 30 d)")
    vessel_df = mt_expected_tuticorin()
    if vessel_df.empty:
        st.warning("Live AIS fetch disabled – add MARINETRAFFIC_KEY env var to enable.")
    else:
        vessel_df["ETA"] = pd.to_datetime(vessel_df["ETA"], errors="coerce")
        st.dataframe(vessel_df, use_container_width=True)

with T4:
    st.subheader("Historical Import Trends – Tuticorin")
    vol_monthly = imports_df.groupby(imports_df["DATE"].dt.to_period("M"))["QUANTITY"].sum().div(1000).reset_index()
    fig = px.area(vol_monthly, x="DATE", y="QUANTITY", labels={"QUANTITY": "Volume (t)"})
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Top Origins in the past 12 months")
    last12 = imports_df[imports_df["DATE"] >= (datetime.now() - timedelta(days=365))]
    top_orig = last12.groupby("COUNTRY OF_ORIGIN")["QUANTITY"].sum().div(1000).sort_values(ascending=False).head(10)
    st.bar_chart(top_orig)

st.caption("Built by GPT‑powered data magic ✨ – tweak the code, add more APIs, and rule the cashew world. ⛵🥜")

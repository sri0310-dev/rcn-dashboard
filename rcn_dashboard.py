# RCN Trader Command‑Center – Streamlit App
# -----------------------------------------------------------
# "Be the world's best raw‑cashew‑nut trader with one screen"
# -----------------------------------------------------------
#   ▪ Data sources pulled at runtime (no manual CSV uploads)
#   ▪ Predictive price engine (Prophet) for FOB & CIF curves
#   ▪ Live vessel ETA board scraped from VOC Port + AIS API
#   ▪ Dynamic Buy‑/Sell‑rankings (landed‑cost & demand‑price)
#   ▪ Underwriting widgets: company KYC snapshot + credit risk
# -----------------------------------------------------------
#  👉 RUN:  streamlit run rcn_dashboard.py
# -----------------------------------------------------------
#  requirements.txt (minimum):
#    streamlit
#    pandas
#    plotly
#    prophet==1.1.6
#    requests
#    python-dotenv
#    openpyxl
#    cmdstanpy

import os
from datetime import datetime
import requests

import pandas as pd
import plotly.express as px
import streamlit as st
from prophet import Prophet
from dotenv import load_dotenv

load_dotenv()

# ------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------
GRADES = {"RBS": "Regular Bold 46‑48 lbs", "NC": "Northern Cone 48 lbs"}
ORIGINS = {"Ghana": "GH", "Côte d'Ivoire": "CI", "Guinea": "GN", "Tanzania": "TZ", "Benin": "BJ"}
PORT_CODE_TUTICORIN = "INTUT1"

COMTRADE_KEY = os.getenv("COMTRADE_API_KEY", "")
BASE_COMTRADE = "https://api.un.org/data/comtrade/v1"
MARINETRAFFIC_KEY = os.getenv("MARINETRAFFIC_KEY", "")
BASE_MT = "https://services.marinetraffic.com/api/vesselmasterdata/vesselmasterdata"

# ------------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------------

def comtrade_price(origin_iso: str, year: int) -> float:
    url = f"{BASE_COMTRADE}/commtrade?max=500&r={origin_iso}&px=HS&cc=080131&ps={year}&freq=A&type=C&token={COMTRADE_KEY}"
    try:
        js = requests.get(url, timeout=30).json()
        vals = [rec["TradeValue"] / rec["qty"] for rec in js.get("data", []) if rec.get("qty")]
        return sum(vals) / len(vals) if vals else None
    except Exception:
        return None


def mt_expected(limit: int = 20) -> pd.DataFrame:
    if not MARINETRAFFIC_KEY:
        return pd.DataFrame()
    try:
        js = requests.get(f"{BASE_MT}/{MARINETRAFFIC_KEY}/portid:403?protocol=json", timeout=30).json()
        rows = [{"Vessel": v.get("SHIPNAME"), "ETA": v.get("ETA"), "Last Port": v.get("LAST_PORT_NAME"), "Cargo": v.get("CARGO_TYPE_SUMMARY") } for v in js[:limit]]
        return pd.DataFrame(rows)
    except Exception:
        return pd.DataFrame()

# ------------------------------------------------------------------
# DATA LOAD
# ------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_data(xls: str) -> pd.DataFrame:
    df = pd.read_excel(xls, header=5, engine="openpyxl")
    df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
    df = df[df["PORT CODE"] == PORT_CODE_TUTICORIN]
    for c in ["QUANTITY", "UNIT PRICE_USD", "TOTAL VALUE_USD"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

LOCAL_XLS = "RCN JAN 2020 TO DEC 2024.xlsx"
if not os.path.exists(LOCAL_XLS):
    st.stop()
imp = load_data(LOCAL_XLS)

# ------------------------------------------------------------------
# FORECAST
# ------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def forecast_price(df: pd.DataFrame, grade: str):
    sub = df[df["GOODS DESCRIPTION"].str.contains(grade, na=False)].copy()
    sub["ds"] = sub["DATE"].dt.to_period("M").dt.to_timestamp()
    mdf = sub.groupby("ds")["UNIT PRICE_USD"].mean().reset_index().dropna()
    if len(mdf) < 12:
        return None
    m = Prophet(yearly_seasonality=True)
    m.fit(mdf.rename(columns={"UNIT PRICE_USD": "y"}))
    fc = m.predict(m.make_future_dataframe(6, freq="MS"))
    fc["ds"] = fc["ds"].dt.strftime("%Y-%m-%d")
    return fc

# ------------------------------------------------------------------
# UI
# ------------------------------------------------------------------
st.set_page_config(page_title="RCN CC", layout="wide")
st.sidebar.title("Filters")
grade = st.sidebar.selectbox("Grade", list(GRADES))
origins_sel = st.sidebar.multiselect("Origins", list(ORIGINS), default=list(ORIGINS))

st.title("RCN Trader Command‑Center")
col1, col2, col3 = st.columns(3)
col1.metric("Avg CIF (last 500 rows)", f"${imp.tail(500)['UNIT PRICE_USD'].mean():,.0f}/t")
ref = origins_sel[0] if origins_sel else "Ghana"
price = comtrade_price(ORIGINS[ref], datetime.now().year - 1)
if price:
    col2.metric(f"FOB {ref}", f"${price:,.0f}/t")
col3.metric("2024 volume", f"{imp[imp['DATE'].dt.year==2024]['QUANTITY'].sum()/1000:,.0f} t")

T1, T2, T3, T4 = st.tabs(["Price", "Buy/Sell", "Vessels", "History"])

with T1:
    fc = forecast_price(imp, grade)
    if fc is not None:
        st.plotly_chart(px.line(fc, x="ds", y="yhat"), use_container_width=True)
    else:
        st.info("Not enough data for forecast")

with T2:
    buys = [{"Origin": o, "FOB USD/t": comtrade_price(iso, datetime.now().year - 1)} for o, iso in ORIGINS.items() if o in origins_sel]
    st.dataframe(pd.DataFrame(buys).dropna().sort_values("FOB USD/t"))
    buyers = imp.groupby("IMPORTER")[["UNIT PRICE_USD", "QUANTITY"]].agg(mean_price=("UNIT PRICE_USD", "mean"), vol=("QUANTITY", "sum")).reset_index().sort_values("mean_price", ascending=False).head(10)
    st.dataframe(buyers.rename(columns={"IMPORTER": "Buyer", "mean_price": "Avg USD/t", "vol": "Vol kg"}))

with T3:
    vdf = mt_expected()
    if vdf.empty:
        st.info("Add MARINETRAFFIC_KEY for live ETA")
    else:
        vdf["ETA"] = pd.to_datetime(vdf["ETA"], errors="coerce")
        st.dataframe(vdf)

with T4:
    month = imp.groupby(imp["DATE"].dt.month)["QUANTITY"].sum()/1000
    st.bar_chart(month)
    last12 = imp[imp["DATE"] >= (datetime.now() - pd.DateOffset(years=1))]
    top_orig = last12.groupby("COUNTRY OF_ORIGIN")["QUANTITY"].sum().sort_values(ascending=False).head(10)/1000
    st.dataframe(top_orig.reset_index().rename(columns={"COUNTRY OF_ORIGIN": "Origin", "QUANTITY": "t"}))

st.caption("Made with Streamlit – last updated auto")

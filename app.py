import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from mlxtend.frequent_patterns import apriori, association_rules
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from datetime import timedelta
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import warnings

warnings.filterwarnings("ignore")

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Mithas Intelligence 7.0", layout="wide")

# ---------------- DATA LOADER ----------------
@st.cache_data
def load_data(file):
    df = pd.read_excel(file)

    col_map = {
        'Invoice No.': 'OrderID',
        'Item Name': 'ItemName',
        'Qty.': 'Quantity',
        'Final Total': 'TotalAmount',
        'Price': 'UnitPrice',
        'Category': 'Category',
        'Timestamp': 'Time',
        'Date': 'Date'
    }
    df = df.rename(columns=col_map)

    for c in ['Quantity', 'TotalAmount', 'UnitPrice']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)

    # --- DATE FIX (CRITICAL) ---
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df[df['Date'].notna()]
        df['DayOfWeek'] = df['Date'].dt.day_name()

    # --- HOUR FIX (ROBUST) ---
    if 'Time' in df.columns:
        df['Hour'] = pd.to_datetime(df['Time'], errors='coerce').dt.hour
        df['Hour'] = df['Hour'].fillna(-1).astype(int)
        df = df[df['Hour'] >= 0]

    return df

# ---------------- PEAK HOUR FOR COMBO ----------------
def get_peak_hour_for_pair(df, item_a, item_b):
    orders_a = set(df[df['ItemName'] == item_a]['OrderID'])
    orders_b = set(df[df['ItemName'] == item_b]['OrderID'])
    common = orders_a & orders_b
    if not common:
        return "N/A"

    subset = df[df['OrderID'].isin(common)]
    peak = subset.groupby('Hour')['Quantity'].sum()
    if peak.empty:
        return "N/A"

    h = int(peak.idxmax())
    return f"{h:02d}:00 - {h+1:02d}:00"

# ---------------- COMBO ANALYSIS ----------------
@st.cache_data
def get_combo_analysis_full(df):
    # HARD MEMORY GUARD
    if df['ItemName'].nunique() > 200:
        return pd.DataFrame()

    top_items = df.groupby('ItemName')['Quantity'].sum().nlargest(150).index
    df = df[df['ItemName'].isin(top_items)]

    basket = (
        df.groupby(['OrderID', 'ItemName'])['Quantity']
        .sum()
        .unstack()
        .fillna(0)
        .astype(int)
    )

    basket = basket.applymap(lambda x: 1 if x > 0 else 0)
    frequent = apriori(basket, min_support=0.005, use_colnames=True)

    if frequent.empty:
        return pd.DataFrame()

    rules = association_rules(frequent, metric="lift", min_threshold=1.05)
    rules['Item A'] = rules['antecedents'].apply(lambda x: list(x)[0])
    rules['Item B'] = rules['consequents'].apply(lambda x: list(x)[0])

    rules['pair'] = rules.apply(lambda x: tuple(sorted([x['Item A'], x['Item B']])), axis=1)
    rules = rules.drop_duplicates('pair')

    item_cat = df.set_index('ItemName')['Category'].to_dict()
    rules['Category A'] = rules['Item A'].map(item_cat)
    rules['Category B'] = rules['Item B'].map(item_cat)

    rules['Specific Item Combo'] = rules['Item A'] + " + " + rules['Item B']

    total_orders = df['OrderID'].nunique()
    rules['Times Sold Together'] = (rules['support'] * total_orders).astype(int)
    rules['Peak Hour'] = rules.apply(
        lambda x: get_peak_hour_for_pair(df, x['Item A'], x['Item B']), axis=1
    )

    return rules

def get_part3_strategy(rules_df):
    if rules_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    proven = rules_df.sort_values('Times Sold Together', ascending=False).head(10).copy()
    proven['Strategy'] = "Bundle & Upsell"

    potential = rules_df[~rules_df.index.isin(proven.index)]
    potential = potential[potential['lift'] > 1.5].sort_values('lift', ascending=False).head(10).copy()
    potential['Strategy'] = "Promote Aggressively"

    return proven, potential

# ---------------- OVERVIEW METRICS ----------------
def get_overview_metrics(df):
    order_totals = df.groupby('OrderID')['TotalAmount'].sum()

    total_rev = order_totals.sum()
    total_orders = order_totals.count()
    num_days = df['Date'].nunique()

    avg_day = total_rev / num_days if num_days else 0
    avg_week = total_rev / max(1, num_days / 7)
    aov = order_totals.mean() if total_orders else 0

    return total_rev, total_orders, avg_day, avg_week, aov

# ---------------- FORECAST ----------------
@st.cache_data
def advanced_forecast(df):
    daily = df.groupby(['Date', 'ItemName'])['Quantity'].sum().reset_index()
    results = []

    top_items = (
        df.groupby('ItemName')['Quantity']
        .sum()
        .nlargest(20)
        .index
    )

    for item in top_items:
        series = daily[daily['ItemName'] == item].set_index('Date')['Quantity']
        series = series.asfreq('D')
        series = series.fillna(method='ffill').fillna(0)

        try:
            if len(series) > 14:
                model = ExponentialSmoothing(
                    series, trend='add', seasonal='add', seasonal_periods=7
                ).fit()
                pred = model.forecast(30)
            else:
                pred = pd.Series([series.mean()] * 30)

            results.append({
                'ItemName': item,
                'Predicted Demand (30 Days)': round(pred.sum(), 0)
            })
        except:
            continue

    return pd.DataFrame(results)

# ---------------- UI ----------------
st.title("📊 Mithas Restaurant Intelligence 7.0")
file = st.sidebar.file_uploader("Upload Monthly Data", type=["xlsx"])

if not file:
    st.info("👋 Upload data to begin.")
    st.stop()

df = load_data(file)

# --- OVERVIEW ---
rev, orders, avg_day, avg_week, aov = get_overview_metrics(df)
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("💰 Total Revenue", f"₹{rev:,.0f}")
c2.metric("🧾 Orders", orders)
c3.metric("📅 Avg / Day", f"₹{avg_day:,.0f}")
c4.metric("🗓️ Avg / Week", f"₹{avg_week:,.0f}")
c5.metric("💳 AOV", f"₹{aov:,.0f}")

st.divider()

# --- COMBOS ---
rules_df = get_combo_analysis_full(df)
proven_df, potential_df = get_part3_strategy(rules_df)

st.subheader("🍔 Smart Combos")
if not rules_df.empty:
    st.dataframe(
        rules_df[['Specific Item Combo', 'Times Sold Together', 'Peak Hour', 'lift']]
        .sort_values('Times Sold Together', ascending=False),
        use_container_width=True,
        hide_index=True
    )
else:
    st.warning("No meaningful combos found.")

# --- FORECAST ---
st.subheader("🔮 Demand Forecast (Top Items)")
forecast_df = advanced_forecast(df)
if not forecast_df.empty:
    st.dataframe(forecast_df, use_container_width=True, hide_index=True)

# --- AI CHAT ---
st.subheader("🤖 Manager Chat")
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input("Ask about sales, combos, demand..."):
    st.chat_message("user").write(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    try:
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            api_key=st.secrets["OPENAI_API_KEY"]
        )
        resp = llm.invoke([
            SystemMessage(content="You are a restaurant data analyst."),
            HumanMessage(content=prompt)
        ])
        st.chat_message("assistant").write(resp.content)
        st.session_state.messages.append({"role": "assistant", "content": resp.content})
    except:
        st.error("⚠️ OpenAI API key issue")

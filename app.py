import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori, association_rules
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from datetime import timedelta
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import warnings

warnings.filterwarnings('ignore')

# --- PAGE CONFIG ---
st.set_page_config(page_title="Mithas Intelligence 7.1", layout="wide")

# --- DATA PROCESSING ---
@st.cache_data
def load_data(file):
    df = pd.read_excel(file)
    col_map = {
        'Invoice No.': 'OrderID', 'Item Name': 'ItemName', 'Qty.': 'Quantity',
        'Final Total': 'TotalAmount', 'Price': 'UnitPrice', 'Category': 'Category',
        'Timestamp': 'Time', 'Date': 'Date'
    }
    df = df.rename(columns=col_map)
    for c in ['Quantity', 'TotalAmount', 'UnitPrice']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df['DayOfWeek'] = df['Date'].dt.day_name()
    
    if 'Time' in df.columns:
        try:
            df['Hour'] = pd.to_datetime(df['Time'].astype(str), format='%H:%M:%S', errors='coerce').dt.hour
            if df['Hour'].isnull().all():
                 df['Hour'] = pd.to_datetime(df['Time'], errors='coerce').dt.hour
        except:
            df['Hour'] = 0
    return df

# --- HELPER FUNCTIONS ---

def get_peak_hour_for_pair(df, item_a, item_b):
    orders_a = set(df[df['ItemName'] == item_a]['OrderID'])
    orders_b = set(df[df['ItemName'] == item_b]['OrderID'])
    common_orders = list(orders_a.intersection(orders_b))
    if not common_orders: return "N/A"
    subset = df[df['OrderID'].isin(common_orders)]
    if 'Hour' not in subset.columns: return "N/A"
    peak = subset['Hour'].mode()
    if not peak.empty:
        p = int(peak[0])
        return f"{p:02d}:00 - {p+1:02d}:00"
    return "N/A"

def get_hourly_details(df):
    """Filters data for 9 AM - 11 PM and aggregates items"""
    if 'Hour' not in df.columns: return pd.DataFrame()
    
    # Filter 9 AM (9) to 11 PM (23)
    mask = (df['Hour'] >= 9) & (df['Hour'] <= 23)
    filtered = df[mask].copy()
    
    # Group by Hour and Item
    hourly_stats = filtered.groupby(['Hour', 'ItemName']).agg({
        'Quantity': 'sum',
        'TotalAmount': 'sum'
    }).reset_index()
    
    # Format Hour String (e.g., "09:00 - 10:00")
    hourly_stats['Time Slot'] = hourly_stats['Hour'].apply(lambda h: f"{int(h):02d}:00 - {int(h)+1:02d}:00")
    
    # Sort by Hour then by Quantity (High to Low)
    hourly_stats = hourly_stats.sort_values(['Hour', 'Quantity'], ascending=[True, False])
    
    return hourly_stats[['Time Slot', 'ItemName', 'Quantity', 'TotalAmount']]

# --- COMBO LOGIC ---

def get_combo_analysis_full(df):
    basket = (df.groupby(['OrderID', 'ItemName'])['Quantity'].sum().unstack().reset_index().fillna(0).set_index('OrderID'))
    basket_sets = basket.applymap(lambda x: 1 if x > 0 else 0)
    frequent = apriori(basket_sets, min_support=0.005, use_colnames=True)
    if frequent.empty: return pd.DataFrame()
    rules = association_rules(frequent, metric="lift", min_threshold=1.05)
    rules['Item A'] = rules['antecedents'].apply(lambda x: list(x)[0])
    rules['Item B'] = rules['consequents'].apply(lambda x: list(x)[0])
    rules['pair'] = rules.apply(lambda x: tuple(sorted([x['Item A'], x['Item B']])), axis=1)
    rules = rules.drop_duplicates(subset='pair')
    item_cat_map = df.set_index('ItemName')['Category'].to_dict()
    rules['Category A'] = rules['Item A'].map(item_cat_map)
    rules['Category B'] = rules['Item B'].map(item_cat_map)
    rules['Specific Item Combo'] = rules['Item A'] + " + " + rules['Item B']
    total_orders = df['OrderID'].nunique()
    rules['Times Sold Together'] = (rules['support'] * total_orders).astype(int)
    rules['Peak Hour'] = rules.apply(lambda x: get_peak_hour_for_pair(df, x['Item A'], x['Item B']), axis=1)
    return rules

def get_part3_strategy(rules_df):
    if rules_df.empty: return pd.DataFrame(), pd.DataFrame()
    proven = rules_df.sort_values('Times Sold Together', ascending=False).head(10).copy()
    potential = rules_df[~rules_df.index.isin(proven.index)]
    potential = potential[potential['lift'] > 1.5].sort_values('lift', ascending=False).head(10).copy()
    return proven, potential

# --- ANALYTICS MODULES (EXISTING) ---
def get_overview_metrics(df):
    total_rev = df['TotalAmount'].sum()
    total_orders = df['OrderID'].nunique()
    num_days = df['Date'].nunique()
    avg_rev_day = total_rev / num_days if num_days > 0 else 0
    num_weeks = max(1, num_days / 7)
    avg_rev_week = total_rev / num_weeks
    aov = total_rev / total_orders if total_orders > 0 else 0
    return total_rev, total_orders, avg_rev_day, avg_rev_week, aov

def get_star_items_with_hours(df):
    total_rev = df['TotalAmount'].sum()
    item_stats = df.groupby('ItemName').agg({'TotalAmount': 'sum'}).reset_index()
    item_stats['Contribution %'] = (item_stats['TotalAmount'] / total_rev) * 100
    item_stats = item_stats.sort_values('TotalAmount', ascending=False).head(20)
    peak_hours, peak_qtys = [], []
    for item in item_stats['ItemName']:
        item_data = df[df['ItemName'] == item]
        if 'Hour' in df.columns and not item_data.empty:
            hour_grouped = item_data.groupby('Hour')['Quantity'].sum()
            peak_hours.append(f"{int(hour_grouped.idxmax()):02d}:00 - {int(hour_grouped.idxmax())+1:02d}:00")
            peak_qtys.append(hour_grouped.max())
        else:
            peak_hours.append("N/A")
            peak_qtys.append(0)
    item_stats['Peak Selling Hour'] = peak_hours
    item_stats['Qty Sold (Peak)'] = peak_qtys
    return item_stats

def analyze_peak_hour_items(df):
    if 'Hour' not in df.columns: return pd.DataFrame(), []
    hourly_rev = df.groupby('Hour')['TotalAmount'].sum()
    top_3_hours = hourly_rev.nlargest(3).index.tolist()
    peak_df = df[df['Hour'].isin(top_3_hours)]
    top_items = peak_df.groupby('ItemName')['Quantity'].sum().sort_values(ascending=False).head(10).reset_index()
    top_items.columns = ['Item Name', 'Qty Sold in Peak Hours']
    return top_items, top_3_hours

def get_contribution_lists(df):
    total_rev = df['TotalAmount'].sum()
    cat_df = df.groupby('Category')['TotalAmount'].sum().reset_index()
    cat_df['Contribution'] = (cat_df['TotalAmount'] / total_rev) * 100
    cat_df = cat_df.sort_values('TotalAmount', ascending=False)
    item_df = df.groupby(['Category', 'ItemName'])['TotalAmount'].sum().reset_index()
    item_df['Contribution'] = (item_df['TotalAmount'] / total_rev) * 100
    item_df = item_df.sort_values(['Category', 'TotalAmount'], ascending=[True, False])
    return cat_df, item_df

def analyze_pareto_hierarchical(df):
    item_rev = df.groupby(['Category', 'ItemName'])['TotalAmount'].sum().reset_index()
    total_revenue = item_rev['TotalAmount'].sum()
    item_rev = item_rev.sort_values('TotalAmount', ascending=False)
    item_rev['Cumulative'] = item_rev['TotalAmount'].cumsum()
    item_rev['CumPerc'] = 100 * item_rev['Cumulative'] / total_revenue
    pareto_items = item_rev[item_rev['CumPerc'] <= 80].copy()
    total_unique_items = df['ItemName'].nunique()
    pareto_unique_items = pareto_items['ItemName'].nunique()
    ratio_text = f"**{pareto_unique_items} items** (out of {total_unique_items}) contribute to 80% of your revenue."
    percentage_of_menu = (pareto_unique_items / total_unique_items) * 100
    cat_rev = df.groupby('Category')['TotalAmount'].sum().reset_index()
    cat_rev['CatContrib'] = (cat_rev['TotalAmount'] / total_revenue) * 100
    merged = pd.merge(pareto_items, cat_rev[['Category', 'CatContrib']], on='Category', how='left')
    merged['ItemContrib'] = (merged['TotalAmount'] / total_revenue) * 100
    display_df = merged[['Category', 'CatContrib', 'ItemName', 'ItemContrib', 'TotalAmount']]
    display_df = display_df.sort_values(['CatContrib', 'TotalAmount'], ascending=[False, False])
    return display_df, ratio_text, percentage_of_menu

def plot_time_series_fixed(df):
    categories = df['Category'].unique()
    for cat in categories:
        st.subheader(f"📈 {cat}")
        cat_data = df[df['Category'] == cat]
        top_items = cat_data.groupby('ItemName')['Quantity'].sum().nlargest(5).index.tolist()
        subset = cat_data[cat_data['ItemName'].isin(top_items)]
        daily = subset.groupby(['Date', 'ItemName'])['Quantity'].sum().reset_index()
        if daily.empty: continue
        fig = px.line(daily, x='Date', y='Quantity', color='ItemName', markers=True)
        for item in top_items:
            avg_val = daily[daily['ItemName'] == item]['Quantity'].mean()
            fig.add_hline(y=avg_val, line_dash="dot", line_color="grey", opacity=0.5)
            fig.add_annotation(x=daily['Date'].max(), y=avg_val, text=f"{item}: {avg_val:.1f}", showarrow=False, yshift=10, font=dict(color="red", size=10))
        fig.update_xaxes(dtick="D2", tickformat="%d %b (%a)")
        fig.update_yaxes(matches=None, showticklabels=True)
        st.plotly_chart(fig, use_container_width=True)

def advanced_forecast(df):
    daily = df.groupby(['Date', 'ItemName'])['Quantity'].sum().reset_index()
    top_cats = df.groupby('Category')['TotalAmount'].sum().nlargest(10).index.tolist()
    forecast_results = []
    for cat in top_cats:
        cat_df = df[df['Category'] == cat]
        top_items = cat_df.groupby('ItemName')['Quantity'].sum().nlargest(10).index.tolist()
        for item in top_items:
            item_data = daily[daily['ItemName'] == item].set_index('Date')['Quantity']
            idx = pd.date_range(item_data.index.min(), item_data.index.max())
            item_data = item_data.reindex(idx, fill_value=0)
            try:
                if len(item_data) > 14:
                    model = ExponentialSmoothing(item_data, trend='add', seasonal='add', seasonal_periods=7).fit()
                    pred = model.forecast(30)
                else:
                    pred = pd.Series([item_data.mean()] * 30, index=pd.date_range(item_data.index.max() + timedelta(days=1), periods=30))
                total_expected_demand = pred.sum()
                forecast_results.append({'Category': cat, 'ItemName': item, 'Total Predicted Demand (Next 30 Days)': round(total_expected_demand, 0)})
            except: continue
    return pd.DataFrame(forecast_results)

# --- MAIN APP LAYOUT ---
st.title("📊 Mithas Restaurant Intelligence 7.1")
uploaded_file = st.sidebar.file_uploader("Upload Monthly Data", type=['xlsx'])

if uploaded_file:
    df = load_data(uploaded_file)
    
    tab1, tab_cat, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Overview", "Category Details", "Pareto (Visual)", "Time Series", "Smart Combos", "Demand Forecast", "AI Chat"
    ])

    # --- TAB 1: OVERVIEW (WITH NEW TABLE) ---
    with tab1:
        st.header("🏢 Business Overview")
        
        # Updated Layout Manager to include "Hourly Breakdown"
        with st.expander("🛠️ Reorder Page Layout", expanded=False):
            overview_order = st.multiselect(
                "Select order of sections:",
                ["Metrics", "Graphs", "Hourly Breakdown (9am-11pm)", "Peak Items List", "Contributions", "Star Items"],
                default=["Metrics", "Graphs", "Hourly Breakdown (9am-11pm)", "Peak Items List", "Contributions", "Star Items"]
            )

        # Define Blocks
        def render_metrics():
            rev, orders, avg_day, avg_week, aov = get_overview_metrics(df)
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("💰 Total Revenue", f"₹{rev:,.0f}")
            c2.metric("🧾 Total Orders", orders)
            c3.metric("📅 Avg Rev/Day", f"₹{avg_day:,.0f}")
            c4.metric("🗓️ Avg Rev/Week", f"₹{avg_week:,.0f}")
            c5.metric("💳 Avg Order Value", f"₹{aov:.0f}")
            st.divider()

        def render_graphs():
            g1, g2 = st.columns(2)
            with g1:
                st.subheader("⌚ Peak Hours Graph")
                if 'Hour' in df.columns:
                    hourly = df.groupby('Hour')['TotalAmount'].sum().reset_index()
                    fig_hourly = px.bar(hourly, x='Hour', y='TotalAmount')
                    avg_hourly = hourly['TotalAmount'].mean()
                    fig_hourly.add_hline(y=avg_hourly, line_dash="dash", line_color="red", annotation_text=f"Avg: ₹{avg_hourly:,.0f}", annotation_position="top right")
                    fig_hourly.update_xaxes(tickmode='linear', dtick=1)
                    st.plotly_chart(fig_hourly, use_container_width=True)
                else: st.warning("No Time data found.")
            with g2:
                st.subheader("📅 Peak Days Graph")
                daily_peak = df.groupby('DayOfWeek')['TotalAmount'].sum().reindex(
                    ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']).reset_index()
                st.bar_chart(daily_peak.set_index('DayOfWeek'))
            st.divider()

        def render_hourly_details():
            st.subheader("🕰️ Hourly Sales Breakdown (9:00 AM - 11:00 PM)")
            st.caption("Detailed breakdown of every item sold in each hour block.")
            hourly_df = get_hourly_details(df)
            if not hourly_df.empty:
                st.dataframe(
                    hourly_df,
                    column_config={
                        "Time Slot": st.column_config.TextColumn("Hour", width="medium"),
                        "ItemName": "Item Name",
                        "Quantity": st.column_config.NumberColumn("Units Sold"),
                        "TotalAmount": st.column_config.NumberColumn("Revenue", format="₹%d")
                    },
                    hide_index=True,
                    use_container_
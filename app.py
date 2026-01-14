import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import fpgrowth, apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from datetime import timedelta
import warnings

# --- NEW IMPORTS FOR HYBRID MODEL ---
from prophet import Prophet
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

# --- PAGE CONFIG ---
st.set_page_config(page_title="Mithas Intelligence 10.2", layout="wide")

# --- 1. ROBUST DATA PROCESSING ---
@st.cache_data
def load_data(file):
    try:
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
        
        # Strict Hygiene: Positive Quantity AND Positive Revenue
        df = df[(df['Quantity'] > 0) & (df['TotalAmount'] > 0)]
        
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            df['DayOfWeek'] = df['Date'].dt.day_name()
        
        if 'Time' in df.columns:
            try:
                df['Hour'] = pd.to_datetime(df['Time'].astype(str), format='mixed', errors='coerce').dt.hour
                df['Hour'] = df['Hour'].fillna(0).astype(int)
            except:
                df['Hour'] = 0
        else:
            df['Hour'] = 0
            
        if 'OrderID' in df.columns:
            df['OrderID'] = df['OrderID'].astype(str)
            
        return df
        
    except Exception as e:
        st.error(f"Error processing file: {e}")
        return pd.DataFrame()

# --- HELPER FUNCTIONS ---

def get_pareto_items(df):
    item_rev = df.groupby('ItemName')['TotalAmount'].sum().sort_values(ascending=False).reset_index()
    total_revenue = item_rev['TotalAmount'].sum()
    item_rev['Cumulative'] = item_rev['TotalAmount'].cumsum()
    item_rev['CumPerc'] = 100 * item_rev['Cumulative'] / total_revenue
    return item_rev[item_rev['CumPerc'] <= 80]['ItemName'].tolist()

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
    if 'Hour' not in df.columns: return pd.DataFrame()
    mask = (df['Hour'] >= 9) & (df['Hour'] <= 23)
    filtered = df[mask].copy()
    hourly_stats = filtered.groupby(['Hour', 'ItemName']).agg({
        'Quantity': 'sum',
        'TotalAmount': 'sum'
    }).reset_index()
    hourly_stats['Time Slot'] = hourly_stats['Hour'].apply(lambda h: f"{int(h):02d}:00 - {int(h)+1:02d}:00")
    hourly_stats = hourly_stats.sort_values(['Hour', 'Quantity'], ascending=[True, False])
    return hourly_stats[['Time Slot', 'ItemName', 'Quantity', 'TotalAmount']]

# --- 2. ROBUST FORECASTING ---

class RobustForecaster:
    def __init__(self):
        self.prophet_model = None
        self.xgb_residual = None
        self.is_fitted = False
        
    def fit(self, df_history):
        self.prophet_model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
        try: self.prophet_model.add_country_holidays(country_name='IN')
        except: pass
        self.prophet_model.fit(df_history)
        
        forecast = self.prophet_model.predict(df_history)
        residuals = df_history['y'].values - forecast['yhat'].values
        
        df_feat = self._create_deterministic_features(df_history)
        self.xgb_residual = XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1)
        self.xgb_residual.fit(df_feat, residuals)
        self.is_fitted = True
        
    def predict(self, periods=30):
        if not self.is_fitted: raise Exception("Model not fitted.")
        
        future_prophet = self.prophet_model.make_future_dataframe(periods=periods, freq='D')
        forecast_prophet = self.prophet_model.predict(future_prophet)
        
        future_feat = self._create_deterministic_features(forecast_prophet[['ds']])
        predicted_residuals = self.xgb_residual.predict(future_feat)
        
        final_prediction = forecast_prophet['yhat'] + predicted_residuals
        final_prediction = final_prediction.apply(lambda x: max(0, x))
        
        result = pd.DataFrame({
            'ds': forecast_prophet['ds'],
            'Predicted_Demand': final_prediction,
            'Lower_Bound': forecast_prophet['yhat_lower'].apply(lambda x: max(0, x)),
            'Upper_Bound': forecast_prophet['yhat_upper'],
            'Trend_Component': forecast_prophet['yhat']
        })
        return result.tail(periods)

    def _create_deterministic_features(self, df_dates):
        X = pd.DataFrame()
        X['dayofweek'] = df_dates['ds'].dt.dayofweek
        X['quarter'] = df_dates['ds'].dt.quarter
        X['month'] = df_dates['ds'].dt.month
        X['is_weekend'] = X['dayofweek'].apply(lambda x: 1 if x >= 5 else 0)
        return X

# --- 3. ADVANCED ASSOCIATION (FIXED) ---

@st.cache_data
def run_advanced_association_cached(df, level='ItemName', min_sup=0.005, min_conf=0.1):
    valid_df = df[(df['Quantity'] > 0) & (df['TotalAmount'] > 0)].copy()
    
    # Create Binary Basket
    basket = valid_df.groupby(['OrderID', level]).size().unstack(fill_value=0)
    basket_bool = (basket > 0)
    
    frequent_itemsets = fpgrowth(basket_bool, min_support=min_sup, use_colnames=True)
    
    if frequent_itemsets.empty: 
        return pd.DataFrame()
    
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_conf)
    
    # Convert sets to list->string for display
    rules['Antecedent'] = rules['antecedents'].apply(lambda x: list(x)[0])
    rules['Consequent'] = rules['consequents'].apply(lambda x: list(x)[0])
    rules['Support (%)'] = rules['support'] * 100
    
    if level == 'Category':
        rules = rules[rules['Antecedent'] != rules['Consequent']]
        
    # Deduplicate rules
    rules['pair_key'] = rules.apply(lambda x: frozenset([x['Antecedent'], x['Consequent']]), axis=1)
    rules = rules.drop_duplicates(subset=['pair_key'])
    
    # --- ERROR FIX: TOTAL QTY CALCULATION ---
    # Ensure rules is a clean dataframe to prevent SettingWithCopy warnings or Index errors
    rules = rules.copy()
    
    # Calculate Total Quantity Sum (Antecedent Qty + Consequent Qty) for valid orders
    qty_matrix = valid_df.groupby(['OrderID', level])['Quantity'].sum().unstack(fill_value=0)
    
    def calc_split_qty(row):
        ant, con = row['Antecedent'], row['Consequent']
        # Check if columns exist in matrix (safety check)
        if ant not in qty_matrix.columns or con not in qty_matrix.columns:
            return "0 (0+0)"
            
        # Boolean mask for orders containing BOTH items
        mask = (qty_matrix[ant] > 0) & (qty_matrix[con] > 0)
        
        # Sum quantities
        sum_ant = qty_matrix.loc[mask, ant].sum()
        sum_con = qty_matrix.loc[mask, con].sum()
        total = sum_ant + sum_con
        return f"{int(total)} ({int(sum_ant)} + {int(sum_con)})"

    rules['Total Item Qty'] = rules.apply(calc_split_qty, axis=1)
    # ----------------------------------------
    
    # Zhangs Metric
    def zhangs_metric(rule):
        sup = rule['support']
        sup_a = rule['antecedent support']
        sup_c = rule['consequent support']
        numerator = sup - (sup_a * sup_c)
        denominator = max(sup * (1 - sup_a), sup_a * (sup_c - sup))
        if denominator == 0: return 0
        return numerator / denominator
    
    rules['zhang'] = rules.apply(zhangs_metric, axis=1)
    
    return rules.sort_values('lift', ascending=False)

@st.cache_data
def get_combo_analysis_full(df):
    return run_advanced_association_cached(df, level='ItemName', min_sup=0.005, min_conf=0.05)

def get_combo_values(df, rules_df):
    if rules_df.empty: return rules_df
    
    item_prices = df.groupby('ItemName').apply(
        lambda x: (x['TotalAmount'].sum() / x['Quantity'].sum()) if x['Quantity'].sum() > 0 else 0
    ).to_dict()
    
    rules_df['Combo Value'] = rules_df['Antecedent'].map(item_prices).fillna(0) + \
                              rules_df['Consequent'].map(item_prices).fillna(0)
    
    rules_df = rules_df.rename(columns={
        'Antecedent': 'Item A',
        'Consequent': 'Item B',
        'Total Item Qty': 'Times Sold Together (Qty)'
    })
    
    # Parse integer for sorting
    rules_df['SortQty'] = rules_df['Times Sold Together (Qty)'].apply(
        lambda x: int(x.split(' ')[0]) if isinstance(x, str) else 0
    )
    
    return rules_df

# --- ANALYTICS MODULES ---
def get_overview_metrics(df):
    total_rev = df['TotalAmount'].sum()
    total_orders = df['OrderID'].nunique()
    num_days = df['Date'].nunique()
    avg_rev_day = total_rev / num_days if num_days > 0 else 0
    num_weeks = max(1, num_days / 7)
    avg_rev_week = total_rev / num_weeks
    aov = total_rev / total_orders if total_orders > 0 else 0
    return total_rev, total_orders, avg_rev_day, avg_rev_week, aov

def get_star_items_with_hours(df, limit_n):
    total_rev = df['TotalAmount'].sum()
    item_stats = df.groupby('ItemName').agg({'TotalAmount': 'sum'}).reset_index()
    item_stats['Contribution %'] = (item_stats['TotalAmount'] / total_rev) * 100
    item_stats = item_stats.sort_values('TotalAmount', ascending=False).head(limit_n)
    peak_hours, peak_qtys = [], []
    for item in item_stats['ItemName']:
        item_data = df[df['ItemName'] == item]
        if 'Hour' in df.columns and not item_data.empty:
            hour_grouped = item_data.groupby('Hour')['Quantity'].sum()
            if not hour_grouped.empty:
                peak_hours.append(f"{int(hour_grouped.idxmax()):02d}:00 - {int(hour_grouped.idxmax())+1:02d}:00")
                peak_qtys.append(hour_grouped.max())
            else:
                peak_hours.append("N/A"); peak_qtys.append(0)
        else:
            peak_hours.append("N/A"); peak_qtys.append(0)
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
    return cat_df

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

def plot_time_series_fixed(df, pareto_list, n_items):
    categories = df['Category'].unique()
    for cat in categories:
        st.subheader(f"📈 {cat}")
        cat_data = df[df['Category'] == cat]
        top_items = cat_data.groupby('ItemName')['TotalAmount'].sum().sort_values(ascending=False).head(n_items).index.tolist()
        subset = cat_data[cat_data['ItemName'].isin(top_items)]
        daily = subset.groupby(['Date', 'ItemName'])['Quantity'].sum().reset_index()
        if daily.empty: continue
        daily['Legend Name'] = daily['ItemName'].apply(lambda x: f"★ {x}" if x in pareto_list else x)
        fig = px.line(daily, x='Date', y='Quantity', color='Legend Name', markers=True)
        st.plotly_chart(fig, use_container_width=True)

# --- MAIN APP LAYOUT ---
st.title("📊 Mithas Restaurant Intelligence 10.2")
uploaded_file = st.sidebar.file_uploader("Upload Monthly Data (Sidebar)", type=['xlsx'])

if uploaded_file:
    df = load_data(uploaded_file)
    
    if df.empty:
        st.error("Uploaded file is empty or invalid.")
        st.stop()

    # Data Health Check
    min_date = df['Date'].min().date()
    max_date = df['Date'].max().date()
    days_data = (max_date - min_date).days + 1
    st.sidebar.info(f"**Data Range:** {days_data} Days")

    # --- CRITICAL FIX: DEFINE VARIABLES BEFORE USE ---
    pareto_list = get_pareto_items(df)
    pareto_count = len(pareto_list) # <--- Fixed NameError
    
    # Pre-calc Smart Combos
    raw_rules = get_combo_analysis_full(df)
    if not raw_rules.empty:
        combo_df = get_combo_values(df, raw_rules)
        proven_df = combo_df.sort_values('SortQty', ascending=False).head(10)
        potential_df = combo_df[~combo_df.index.isin(proven_df.index)]
        potential_df = potential_df[potential_df['lift'] > 1.5].sort_values('lift', ascending=False).head(10)
        proven_pairs = set(zip(proven_df['Item A'], proven_df['Item B']))
        potential_pairs = set(zip(potential_df['Item A'], potential_df['Item B']))
    else:
        combo_df, proven_df, potential_df = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        proven_pairs, potential_pairs = set(), set()

    # Tabs
    tab1, tab_day_wise, tab_cat, tab2, tab3, tab4, tab_assoc, tab5, tab6 = st.tabs([
        "Overview", "Day wise Analysis", "Category Details", "Pareto (Visual)", "Time Series", "Smart Combos", "Association Analysis", "Demand Forecast", "AI Chat"
    ])

    # [TAB 1: OVERVIEW]
    with tab1:
        st.header("🏢 Business Overview")
        overview_order = st.multiselect("Sections", ["Metrics", "Graphs", "Hourly", "Peak Items", "Contributions", "Star Items"], default=["Metrics", "Graphs", "Hourly", "Peak Items", "Contributions", "Star Items"])
        
        def render_metrics():
            rev, orders, avg_day, avg_week, aov = get_overview_metrics(df)
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("💰 Revenue", f"₹{rev:,.0f}"); c2.metric("🧾 Orders", orders)
            c3.metric("📅 Daily Avg", f"₹{avg_day:,.0f}"); c4.metric("🗓️ Weekly Avg", f"₹{avg_week:,.0f}")
            c5.metric("💳 AOV", f"₹{aov:.0f}"); st.divider()

        def render_graphs():
            g1, g2 = st.columns(2)
            with g1:
                hourly = df.groupby('Hour')['TotalAmount'].sum().reset_index()
                st.plotly_chart(px.bar(hourly, x='Hour', y='TotalAmount', title="Peak Hours"), use_container_width=True)
            with g2:
                daily = df.groupby('DayOfWeek')['TotalAmount'].sum().reindex(['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']).reset_index()
                st.plotly_chart(px.bar(daily, x='DayOfWeek', y='TotalAmount', title="Peak Days"), use_container_width=True)
            st.divider()

        def render_hourly_aggregated():
            st.subheader("🕰️ Hourly Details")
            hourly_df = get_hourly_details(df)
            if not hourly_df.empty:
                hourly_df['ItemName'] = hourly_df['ItemName'].apply(lambda x: f"★ {x}" if x in pareto_list else x)
                st.dataframe(hourly_df, use_container_width=True, height=300)
            else: st.info("No data.")
            st.divider()

        def render_peak_list():
            l1, l2 = st.columns(2)
            with l1:
                peak_items_df, top_hrs = analyze_peak_hour_items(df)
                st.subheader(f"🔥 Peak Hour Items {top_hrs}")
                st.dataframe(peak_items_df, hide_index=True, use_container_width=True)
            with l2:
                top_days = df.groupby('Date')['TotalAmount'].sum().sort_values(ascending=False).head(5).reset_index()
                top_days['Date'] = top_days['Date'].dt.strftime('%Y-%m-%d')
                st.subheader("💰 Top Revenue Days")
                st.dataframe(top_days, hide_index=True, use_container_width=True)
            st.divider()

        def render_contributions():
            st.subheader("📂 Category Share")
            cat_cont = get_contribution_lists(df)
            st.plotly_chart(px.pie(cat_cont, values='TotalAmount', names='Category', hole=0.3), use_container_width=True)
            st.divider()

        def render_star_items():
            st.subheader("⭐ Top Star Items")
            slider_max = max(10, pareto_count) # Safe now
            n_star = st.slider("Count", 10, slider_max, 20)
            star_df = get_star_items_with_hours(df, n_star)
            star_df['Item Name'] = star_df['ItemName'].apply(lambda x: f"★ {x}" if x in pareto_list else x)
            st.dataframe(star_df, use_container_width=True)

        block_map = {"Metrics": render_metrics, "Graphs": render_graphs, "Hourly": render_hourly_aggregated, "Peak Items": render_peak_list, "Contributions": render_contributions, "Star Items": render_star_items}
        for b in overview_order: block_map[b]()

    # [TAB 2 & 3: RESTORED CONTENT]
    with tab2:
        st.header("🏆 Pareto Analysis")
        p_df, r_msg, m_perc = analyze_pareto_hierarchical(df)
        st.info(f"💡 {r_msg}")
        p_df['ItemName'] = p_df['ItemName'].apply(lambda x: f"★ {x}")
        st.dataframe(p_df, use_container_width=True)

    with tab3:
        st.header("📅 Daily Trends")
        n_ts = st.slider("Top Items per Category", 3, 20, 5)
        plot_time_series_fixed(df, pareto_list, n_ts)

    # --- TAB: DAY WISE ANALYSIS ---
    with tab_day_wise:
        st.header("📅 Day-wise Deep Dive")
        days = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
        
        st.subheader("⚠️ Items Not Worth Producing (<3 sales in 3-day window)")
        sel_nw_cat = st.selectbox("Filter Category", ["All"] + sorted(df['Category'].unique().tolist()))
        
        nw_dict = {}
        max_l = 0
        for d in days:
            idx = days.index(d)
            win_days = [days[(idx-1)%7], d, days[(idx+1)%7]]
            win_df = df[df['DayOfWeek'].isin(win_days)]
            day_df = df[df['DayOfWeek'] == d]
            
            if sel_nw_cat != "All":
                win_df = win_df[win_df['Category'] == sel_nw_cat]
                day_df = day_df[day_df['Category'] == sel_nw_cat]
            
            sums = win_df.groupby('ItemName')['Quantity'].sum()
            cands = sums[sums < 3].index.tolist()
            actives = day_df[day_df['Quantity'] > 0]['ItemName'].unique().tolist()
            final = [f"★ {x}" if x in pareto_list else x for x in cands if x in actives]
            nw_dict[d] = final
            max_l = max(max_l, len(final))
            
        for d in days: nw_dict[d] += [""] * (max_l - len(nw_dict[d]))
        st.dataframe(pd.DataFrame(nw_dict), use_container_width=True)
        
        st.markdown("---")
        st.subheader("📉 Missed Upsell (Orders < AOV)")
        m_aov = df['TotalAmount'].sum() / df['OrderID'].nunique()
        d_aov = df.groupby(df['Date'].dt.date).apply(lambda x: pd.Series({'Pct': (x['TotalAmount'] < m_aov).mean()*100})).reset_index()
        d_aov.columns = ['Date','Pct']
        fig_aov = px.bar(d_aov, x='Date', y='Pct', color='Pct', color_continuous_scale='RdYlGn_r', title=f"Daily % Orders < Monthly Avg (₹{m_aov:.0f})")
        fig_aov.add_hline(y=50, line_dash="dot", line_color="red")
        st.plotly_chart(fig_aov, use_container_width=True)

    # --- TAB: CATEGORY DETAILS ---
    with tab_cat:
        st.header("📂 Category Deep-Dive")
        s_cat = st.selectbox("Select Category", sorted(df['Category'].unique()))
        c_data = df[df['Category'] == s_cat]
        
        c_agg = c_data.groupby('ItemName').agg({'TotalAmount':'sum', 'Quantity':'sum'}).reset_index().sort_values('TotalAmount', ascending=False)
        def mark(r):
            n = r['ItemName']
            if n in pareto_list: n = f"★ {n}"
            if r['TotalAmount'] < 500: n += " **"
            elif r['TotalAmount'] < 1500: n += " *"
            return n
        c_agg['Item Name'] = c_agg.apply(mark, axis=1)
        st.dataframe(c_agg, use_container_width=True)
        st.caption("Legend: `**` < ₹500 | `*` < ₹1500 | `★` Pareto")
        
        st.divider()
        st.subheader("Hourly Matrix")
        s_day = st.selectbox("Select Day", days)
        m_data = c_data[c_data['DayOfWeek'] == s_day]
        if not m_data.empty:
            piv = m_data.groupby(['ItemName', 'Hour'])['Quantity'].sum().unstack(fill_value=0)
            for h in range(9, 24): 
                if h not in piv.columns: piv[h] = 0
            st.dataframe(piv[sorted(piv.columns)], use_container_width=True)
        else: st.warning("No data.")

    # --- TAB 4: SMART COMBOS ---
    with tab4:
        st.header("🍔 Smart Combos")
        def star_c(r):
            a, b = r['Item A'], r['Item B']
            if a in pareto_list: a = f"★ {a}"
            if b in pareto_list: b = f"★ {b}"
            return f"{a} + {b}"
            
        if not combo_df.empty:
            combo_df['Display'] = combo_df.apply(star_combo, axis=1) # Reuse function from pre-calc logic? No, define local or reuse.
            # Local definition above 'star_c' is fine.
            combo_df['Display'] = combo_df.apply(star_c, axis=1)
            proven_df['Display'] = proven_df.apply(star_c, axis=1)
            potential_df['Display'] = potential_df.apply(star_c, axis=1)
            
            c1, c2 = st.columns(2)
            with c1: st.subheader("🔥 Winners"); st.dataframe(proven_df[['Display', 'Times Sold Together (Qty)', 'Combo Value']], hide_index=True, use_container_width=True)
            with c2: st.subheader("💎 Gems"); st.dataframe(potential_df[['Display', 'lift', 'Combo Value']], hide_index=True, use_container_width=True)
            st.subheader("All Combos"); st.dataframe(combo_df[['Display', 'Times Sold Together (Qty)', 'lift', 'Combo Value']], hide_index=True)
        else: st.warning("No combos found.")

    # --- TAB 5: ASSOCIATION ANALYSIS ---
    with tab_assoc:
        st.header("🧬 Association Rules")
        c1, c2 = st.columns(2)
        lvl = c1.radio("Level", ["ItemName", "Category"], horizontal=True)
        
        # Dynamic Slider
        v_df = df[(df['Quantity'] > 0) & (df['TotalAmount'] > 0)]
        b_glob = v_df.groupby(['OrderID', lvl]).size().unstack(fill_value=0) > 0
        max_s = b_glob.mean().max() * 100 if not b_glob.empty else 1.0
        
        with c2:
            min_s_pct = st.slider("Min Support (%)", 0.01, float(max_s+1), 0.1, step=0.01)
        
        with st.spinner("Analyzing..."):
            rules = run_advanced_association_cached(df, level=lvl, min_sup=min_s_pct/100)
            if not rules.empty:
                def get_stat(r):
                    if lvl == 'Category': return "Category"
                    p = tuple(sorted([r['Antecedent'], r['Consequent']]))
                    if p in proven_pairs: return "🔥 Winner"
                    if p in potential_pairs: return "💎 Gem"
                    return "Normal"
                rules['Status'] = rules.apply(get_stat, axis=1)
                
                if lvl == 'ItemName':
                    rules['Antecedent'] = rules['Antecedent'].apply(lambda x: f"★ {x}" if x in pareto_list else x)
                    rules['Consequent'] = rules['Consequent'].apply(lambda x: f"★ {x}" if x in pareto_list else x)
                
                st.dataframe(rules[['Status', 'Antecedent', 'Consequent', 'Support (%)', 'Total Item Qty', 'confidence', 'lift']], use_container_width=True)
            else: st.warning("No rules found.")

    # --- TAB 7: DEMAND FORECAST ---
    with tab5:
        st.header("🔮 Demand Forecast")
        s_item = st.selectbox("Select Item", sorted(df['ItemName'].unique()))
        if st.button("Forecast"):
            with st.spinner("Forecasting..."):
                i_data = df[df['ItemName'] == s_item].groupby('Date')['Quantity'].sum().reset_index()
                i_data.columns = ['ds', 'y']
                if len(i_data) < 14: st.error("Need 14+ days data.")
                else:
                    try:
                        m = RobustForecaster()
                        m.fit(i_data)
                        fc = m.predict(30)
                        
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=i_data['ds'], y=i_data['y'], mode='markers', name='Actual', marker=dict(color='gray', opacity=0.5)))
                        fig.add_trace(go.Scatter(x=fc['ds'], y=fc['Upper_Bound'], mode='lines', line=dict(width=0), showlegend=False))
                        fig.add_trace(go.Scatter(x=fc['ds'], y=fc['Lower_Bound'], mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(0,200,150,0.2)', name='Confidence'))
                        fig.add_trace(go.Scatter(x=fc['ds'], y=fc['Predicted_Demand'], mode='lines', name='Forecast', line=dict(color='#00CC96')))
                        st.plotly_chart(fig, use_container_width=True)
                        st.dataframe(fc, use_container_width=True)
                    except Exception as e: st.error(str(e))

    with tab6: st.header("AI Chat"); st.info("Chat Interface Active")

else:
    st.info("👋 Please upload data.")
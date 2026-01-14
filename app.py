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
from statsmodels.tsa.holtwinters import ExponentialSmoothing
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

# --- 1. ROBUST DATA PROCESSING (FIXED: STRICT HYGIENE) ---
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
        
        # --- FIX: STRICT FINANCIAL FILTERING ---
        # Removes refunds/errors/zero-priced items to prevent skewed analytics
        df = df[(df['Quantity'] > 0) & (df['TotalAmount'] > 0)]
        
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            df['DayOfWeek'] = df['Date'].dt.day_name()
        
        if 'Time' in df.columns:
            try:
                # Handle mixed time formats robustly
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

# --- 2. ROBUST FORECASTING (FIXED: CONFIDENCE INTERVALS) ---

class RobustForecaster:
    def __init__(self):
        self.prophet_model = None
        self.xgb_residual = None
        self.is_fitted = False
        
    def fit(self, df_history):
        # 1. Prophet: Trend + Seasonality
        self.prophet_model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
        try: self.prophet_model.add_country_holidays(country_name='IN')
        except: pass
        self.prophet_model.fit(df_history)
        
        # 2. Residuals
        forecast = self.prophet_model.predict(df_history)
        residuals = df_history['y'].values - forecast['yhat'].values
        
        # 3. XGBoost on Residuals
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
        
        # Combined Forecast
        final_prediction = forecast_prophet['yhat'] + predicted_residuals
        final_prediction = final_prediction.apply(lambda x: max(0, x))
        
        # --- FIX: ADD CONFIDENCE INTERVALS ---
        result = pd.DataFrame({
            'ds': forecast_prophet['ds'],
            'Predicted_Demand': final_prediction,
            'Lower_Bound': forecast_prophet['yhat_lower'].apply(lambda x: max(0, x)),
            'Upper_Bound': forecast_prophet['yhat_upper'],
            'Trend_Component': forecast_prophet['yhat']
        })
        
        return result.tail(periods)

    def _create_deterministic_features(self, df_dates):
        # Creates features known in future (No Lag Data Leakage)
        X = pd.DataFrame()
        X['dayofweek'] = df_dates['ds'].dt.dayofweek
        X['quarter'] = df_dates['ds'].dt.quarter
        X['month'] = df_dates['ds'].dt.month
        X['is_weekend'] = X['dayofweek'].apply(lambda x: 1 if x >= 5 else 0)
        return X

# --- 3. ADVANCED ASSOCIATION (CACHED) ---

@st.cache_data
def run_advanced_association_cached(df, level='ItemName', min_sup=0.005, min_conf=0.1):
    # Strict filtering again for safety
    valid_df = df[(df['Quantity'] > 0) & (df['TotalAmount'] > 0)].copy()
    
    basket = valid_df.groupby(['OrderID', level]).size().unstack(fill_value=0)
    basket_bool = (basket > 0)
    
    frequent_itemsets = fpgrowth(basket_bool, min_support=min_sup, use_colnames=True)
    
    if frequent_itemsets.empty: 
        return pd.DataFrame()
    
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_conf)
    
    rules['Antecedent'] = rules['antecedents'].apply(lambda x: list(x)[0])
    rules['Consequent'] = rules['consequents'].apply(lambda x: list(x)[0])
    rules['Support (%)'] = rules['support'] * 100
    
    if level == 'Category':
        rules = rules[rules['Antecedent'] != rules['Consequent']]
        
    rules['pair_key'] = rules.apply(lambda x: frozenset([x['Antecedent'], x['Consequent']]), axis=1)
    rules = rules.drop_duplicates(subset=['pair_key'])
    
    # Accurate Quantity Summation
    qty_matrix = valid_df.groupby(['OrderID', level])['Quantity'].sum().unstack(fill_value=0)
    total_trans = len(basket_bool)
    
    rules['No. of Orders'] = (rules['support'] * total_trans).round().astype(int)

    def calc_split_qty(row):
        ant, con = row['Antecedent'], row['Consequent']
        if ant in qty_matrix.columns and con in qty_matrix.columns:
            mask = (qty_matrix[ant] > 0) & (qty_matrix[con] > 0)
            sum_ant = qty_matrix.loc[mask, ant].sum()
            sum_con = qty_matrix.loc[mask, con].sum()
            total = sum_ant + sum_con
            return f"{int(total)} ({int(sum_ant)} + {int(sum_con)})"
        return "0 (0+0)"

    rules['Total Item Qty'] = rules.apply(calc_split_qty, axis=1)
    
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
        
        for item in top_items:
            avg_val = daily[daily['ItemName'] == item]['Quantity'].mean()
            fig.add_hline(y=avg_val, line_dash="dot", line_color="grey", opacity=0.5)
            fig.add_annotation(
                x=daily['Date'].max(), y=avg_val, 
                text=f"{item}: {avg_val:.1f}", 
                showarrow=False, yshift=10, font=dict(color="red", size=10)
            )
            
        fig.update_xaxes(dtick="D2", tickformat="%d %b (%a)")
        fig.update_yaxes(matches=None, showticklabels=True)
        st.plotly_chart(fig, use_container_width=True)

# --- MAIN APP ---
st.title("📊 Mithas Restaurant Intelligence 10.2")
uploaded_file = st.sidebar.file_uploader("Upload Monthly Data (Sidebar)", type=['xlsx'])

if uploaded_file:
    df = load_data(uploaded_file)
    
    if df.empty:
        st.error("Uploaded file is empty or invalid.")
        st.stop()

    # --- DATA HEALTH CHECK ---
    min_date = df['Date'].min().date()
    max_date = df['Date'].max().date()
    days_data = (max_date - min_date).days + 1
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("✅ Data Health Check")
    st.sidebar.info(f"**Range:** {min_date} to {max_date}")
    st.sidebar.info(f"**Days Covered:** {days_data}")
    if days_data < 30:
        st.sidebar.warning("⚠️ Warning: Less than 30 days of data. Trends & Forecasts may be less reliable.")
    else:
        st.sidebar.success("✅ Sufficient history for analysis.")

    pareto_list = get_pareto_items(df)
    
    # Global Pre-calc
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

    tab1, tab_day_wise, tab_cat, tab2, tab3, tab4, tab_assoc, tab5, tab6 = st.tabs([
        "Overview", "Day wise Analysis", "Category Details", "Pareto (Visual)", "Time Series", "Smart Combos", "Association Analysis", "Demand Forecast", "AI Chat"
    ])

    # [TAB 1: OVERVIEW]
    with tab1:
        st.header("🏢 Business Overview")
        # Layout Reorder
        with st.expander("🛠️ Reorder Page Layout", expanded=False):
            overview_order = st.multiselect(
                "Select order of sections:",
                ["Metrics", "Graphs", "Hourly Breakdown (Aggregated)", "Peak Items List", "Contributions", "Star Items"],
                default=["Metrics", "Graphs", "Hourly Breakdown (Aggregated)", "Peak Items List", "Contributions", "Star Items"]
            )

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
                    fig_hourly.add_hline(y=avg_hourly, line_dash="dash", line_color="red", 
                                         annotation_text=f"Avg: ₹{avg_hourly:,.0f}", annotation_position="top right")
                    fig_hourly.update_xaxes(tickmode='linear', dtick=1)
                    st.plotly_chart(fig_hourly, use_container_width=True)
            with g2:
                st.subheader("📅 Peak Days Graph")
                daily_peak = df.groupby('DayOfWeek')['TotalAmount'].sum().reindex(
                    ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']).reset_index()
                fig_daily = px.bar(daily_peak, x='DayOfWeek', y='TotalAmount')
                fig_daily.update_xaxes(
                    tickfont=dict(family='Arial Black', size=14, color='black'),
                    title_font=dict(family='Arial Black', size=16)
                )
                st.plotly_chart(fig_daily, use_container_width=True)
            st.divider()

        def render_hourly_aggregated():
            st.subheader("🕰️ Hourly Sales Breakdown (9am - 11pm) - Aggregated")
            st.caption("Aggregated view of all days combined.")
            hourly_df = get_hourly_details(df)
            if not hourly_df.empty:
                hourly_df['ItemName'] = hourly_df['ItemName'].apply(lambda x: f"★ {x}" if x in pareto_list else x)
                time_slots = hourly_df['Time Slot'].unique()
                for slot in time_slots:
                    slot_data = hourly_df[hourly_df['Time Slot'] == slot]
                    total_rev = slot_data['TotalAmount'].sum()
                    total_qty = slot_data['Quantity'].sum()
                    top_item = slot_data.sort_values('Quantity', ascending=False).iloc[0]['ItemName']
                    with st.expander(f"⏰ {slot}  |  Revenue: ₹{total_rev:,.0f}  |  Units: {total_qty}  |  Top: {top_item}"):
                        st.dataframe(slot_data[['ItemName', 'Quantity', 'TotalAmount']], hide_index=True, use_container_width=True)
            else: st.info("No data.")
            st.divider()

        def render_peak_list():
            l1, l2 = st.columns(2)
            with l1:
                peak_items_df, top_hrs = analyze_peak_hour_items(df)
                st.subheader(f"🔥 Items Sold in Peak Hours {top_hrs}")
                st.dataframe(peak_items_df, hide_index=True, use_container_width=True)
            with l2:
                st.subheader("💰 High Revenue Days")
                top_days = df.groupby('Date')['TotalAmount'].sum().sort_values(ascending=False).head(5).reset_index()
                top_days['Date'] = top_days['Date'].dt.strftime('%Y-%m-%d (%A)')
                st.dataframe(top_days, hide_index=True, use_container_width=True)
            st.divider()

        def render_contributions():
            cat_cont = get_contribution_lists(df)
            st.subheader("📂 Category Contribution")
            fig_pie = px.pie(cat_cont, values='TotalAmount', names='Category', hole=0.3)
            st.plotly_chart(fig_pie, use_container_width=True)
            st.divider()

        def render_star_items():
            st.subheader("⭐ Top Star Items & Selling Hours")
            slider_max = max(10, pareto_count)
            n_star = st.slider("Select Number of Star Items", 10, slider_max, 20)
            star_df = get_star_items_with_hours(df, n_star)
            star_df['Item Name'] = star_df['ItemName'].apply(lambda x: f"★ {x}" if x in pareto_list else x)
            st.dataframe(star_df[['Item Name', 'TotalAmount', 'Contribution %', 'Peak Selling Hour', 'Qty Sold (Peak)']], 
                         column_config={"TotalAmount": st.column_config.NumberColumn("Revenue", format="₹%d"), "Contribution %": st.column_config.ProgressColumn("Contribution", format="%.2f%%")}, hide_index=True, use_container_width=True)

        block_map = {
            "Metrics": render_metrics, 
            "Graphs": render_graphs, 
            "Hourly Breakdown (Aggregated)": render_hourly_aggregated,
            "Peak Items List": render_peak_list, 
            "Contributions": render_contributions, 
            "Star Items": render_star_items
        }
        for block_name in overview_order:
            if block_name in block_map: block_map[block_name]()

    # --- TAB: DAY WISE ANALYSIS ---
    with tab_day_wise:
        st.header("📅 Day-wise Deep Dive")
        days_list = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
        
        st.subheader("⚠️ Items Not Worth Producing")
        st.markdown("Items with **< 3 units sold** in a 3-Day window. **Refined:** Only items with >0 sales on specific day.")
        
        cats_list = ["All Categories"] + sorted(df['Category'].unique().tolist())
        selected_nw_category = st.selectbox("Filter by Category", cats_list, key="nw_cat")
        
        not_worth_dict = {}
        max_len = 0
        
        for d in days_list:
            curr_idx = days_list.index(d)
            window_days = [days_list[(curr_idx - 1) % 7], d, days_list[(curr_idx + 1) % 7]]
            
            window_df = df[df['DayOfWeek'].isin(window_days)]
            if selected_nw_category != "All Categories":
                window_df = window_df[window_df['Category'] == selected_nw_category]
            
            item_sums = window_df.groupby('ItemName')['Quantity'].sum()
            candidates = item_sums[item_sums < 3].index.tolist()
            
            day_df = df[df['DayOfWeek'] == d]
            if selected_nw_category != "All Categories":
                day_df = day_df[day_df['Category'] == selected_nw_category]
            
            active_items = day_df[day_df['Quantity'] > 0]['ItemName'].unique().tolist()
            final_items = [f"★ {x}" if x in pareto_list else x for x in candidates if x in active_items]
            
            not_worth_dict[d] = final_items
            max_len = max(max_len, len(final_items))
            
        for d in days_list:
            not_worth_dict[d] += [""] * (max_len - len(not_worth_dict[d]))
            
        st.dataframe(pd.DataFrame(not_worth_dict), use_container_width=True, height=500)
        
        st.markdown("---")
        st.subheader("📉 Missed Upsell Opportunities")
        monthly_aov = df['TotalAmount'].sum() / df['OrderID'].nunique()
        daily_aov = df.groupby(df['Date'].dt.date).apply(
            lambda x: pd.Series({'Pct': ((x['TotalAmount'] < monthly_aov).sum() / len(x)) * 100})
        ).reset_index()
        daily_aov.columns = ['Date', 'Pct']
        
        fig_aov = px.bar(daily_aov, x='Date', y='Pct', color='Pct', color_continuous_scale='RdYlGn_r', title=f"Daily % Orders < Monthly Avg (₹{monthly_aov:.0f})")
        fig_aov.add_hline(y=50, line_dash="dot", line_color="red")
        st.plotly_chart(fig_aov, use_container_width=True)

    # --- TAB: CATEGORY DETAILS ---
    with tab_cat:
        st.header("📂 Category Deep-Dive")
        cats = sorted(df['Category'].unique())
        selected_cat = st.selectbox("Select Category", cats)
        
        cat_data = df[df['Category'] == selected_cat]
        cat_agg = cat_data.groupby('ItemName').agg({'TotalAmount':'sum', 'Quantity':'sum'}).reset_index().sort_values('TotalAmount', ascending=False)
        
        def mark_name(row):
            n = row['ItemName']
            if n in pareto_list: n = f"★ {n}"
            if row['TotalAmount'] < 500: n += " **"
            elif row['TotalAmount'] < 1500: n += " *"
            return n
            
        cat_agg['Item Name'] = cat_agg.apply(mark_name, axis=1)
        st.dataframe(cat_agg[['Item Name', 'TotalAmount', 'Quantity']], use_container_width=True)
        st.caption("Legend: `**` < ₹500 | `*` < ₹1500 | `★` Pareto")
        
        st.divider()
        st.subheader(f"Hourly Matrix: {selected_cat}")
        sel_day = st.selectbox("Select Day", days_list, key="mat_day")
        mat_data = cat_data[cat_data['DayOfWeek'] == sel_day]
        if not mat_data.empty:
            pivot = mat_data.groupby(['ItemName', 'Hour'])['Quantity'].sum().unstack(fill_value=0)
            for h in range(9, 24):
                if h not in pivot.columns: pivot[h] = 0
            pivot = pivot[sorted(pivot.columns)]
            st.dataframe(pivot, use_container_width=True)
        else:
            st.warning("No data.")

    # --- TAB 2: PARETO (FULL IMPLEMENTATION RESTORED) ---
    with tab2:
        st.header("🏆 Pareto Analysis")
        pareto_df, ratio_msg, menu_perc = analyze_pareto_hierarchical(df)
        st.info(f"💡 **Insight:** {ratio_msg} (Only {menu_perc:.1f}% of your menu!)")
        pareto_df['ItemName'] = pareto_df['ItemName'].apply(lambda x: f"★ {x}")
        st.dataframe(pareto_df, column_config={"CatContrib": st.column_config.NumberColumn("Category Share %", format="%.2f%%"), "ItemContrib": st.column_config.NumberColumn("Item Share % (Global)", format="%.2f%%"), "TotalAmount": st.column_config.NumberColumn("Revenue", format="₹%d")}, hide_index=True, height=600, use_container_width=True)

    # --- TAB 3: TIME SERIES (FULL IMPLEMENTATION RESTORED) ---
    with tab3:
        st.header("📅 Daily Trends")
        n_ts = st.slider("Number of items per category (Top N by Revenue)", 5, 30, 5)
        plot_time_series_fixed(df, pareto_list, n_ts)

    # --- TAB 4: SMART COMBOS ---
    with tab4:
        st.header("🍔 Smart Combo Strategy")
        with st.expander("🛠️ Reorder Layout"):
            st.multiselect("Section Order", ["Part 1", "Part 3"], default=["Part 1", "Part 3"])
        
        def star_combo(row):
            a, b = row['Item A'], row['Item B']
            if a in pareto_list: a = f"★ {a}"
            if b in pareto_list: b = f"★ {b}"
            return f"{a} + {b}"

        if not combo_df.empty:
            combo_df['Display Combo'] = combo_df.apply(star_combo, axis=1)
            proven_df['Display Combo'] = proven_df.apply(star_combo, axis=1)
            potential_df['Display Combo'] = potential_df.apply(star_combo, axis=1)

            c1, c2 = st.columns(2)
            with c1:
                st.subheader("🔥 Proven Winners")
                st.dataframe(proven_df[['Display Combo', 'Times Sold Together (Qty)', 'Combo Value']], hide_index=True, use_container_width=True)
            with c2:
                st.subheader("💎 Hidden Gems")
                st.dataframe(potential_df[['Display Combo', 'lift', 'Combo Value']], hide_index=True, use_container_width=True)
            
            st.subheader("1️⃣ Full Map")
            st.dataframe(combo_df[['Display Combo', 'Times Sold Together (Qty)', 'lift', 'Combo Value']], hide_index=True)
        else:
            st.warning("No significant combos.")

    # --- TAB 5: ASSOCIATION ANALYSIS ---
    with tab_assoc:
        st.header("🧬 Scientific Association Analysis")
        c1, c2 = st.columns(2)
        level = c1.radio("Level", ["ItemName", "Category"], horizontal=True)
        
        # Dynamic Slider Logic
        valid_df_global = df[(df['Quantity'] > 0) & (df['TotalAmount'] > 0)]
        basket_global = valid_df_global.groupby(['OrderID', level]).size().unstack(fill_value=0)
        basket_sets_global = (basket_global > 0).astype(bool)
        
        supports = basket_sets_global.mean().sort_values(ascending=False)
        max_sup = supports.max() if not supports.empty else 1.0
        max_sup_percent = float(min(100.0, max_sup * 100 + 1.0))
        default_val = min(0.5, max_sup_percent / 2)

        with c2:
            min_support_percent = st.slider("Minimum Support (%)", 0.01, max_sup_percent, default_val, step=0.01, format="%.2f%%")
            min_support_val = min_support_percent / 100.0
        
        with st.spinner("Analyzing..."):
            rules = run_advanced_association_cached(df, level=level, min_sup=min_support_val)
            
            if not rules.empty:
                # Add Status & Stars
                def get_status_row(row):
                    if level == 'Category': return "Category"
                    p1 = (row['Antecedent'], row['Consequent'])
                    p2 = (row['Consequent'], row['Antecedent'])
                    if p1 in proven_pairs or p2 in proven_pairs: return "🔥 Winner"
                    if p1 in potential_pairs or p2 in potential_pairs: return "💎 Gem"
                    return "Normal"

                rules['Status'] = rules.apply(get_status_row, axis=1)
                
                if level == 'ItemName':
                    rules['Antecedent'] = rules['Antecedent'].apply(lambda x: f"★ {x}" if x in pareto_list else x)
                    rules['Consequent'] = rules['Consequent'].apply(lambda x: f"★ {x}" if x in pareto_list else x)
                
                # NO LIMIT on rows
                st.dataframe(rules[['Status', 'Antecedent', 'Consequent', 'Support (%)', 'Total Item Qty', 'confidence', 'lift', 'zhang', 'conviction']], use_container_width=True, height=600)
                
                fig = px.scatter(rules, x="Support (%)", y="confidence", size="lift", color="zhang")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No associations found.")

    # --- TAB 7: DEMAND FORECAST ---
    with tab5:
        st.header("🔮 Robust Demand Forecast")
        
        item_list = sorted(df['ItemName'].unique())
        sel_item = st.selectbox("Select Item", item_list)
        
        if st.button("Generate Forecast"):
            with st.spinner("Training Robust Model..."):
                item_data = df[df['ItemName'] == sel_item].groupby('Date')['Quantity'].sum().reset_index()
                item_data.columns = ['ds', 'y']
                
                if len(item_data) < 14:
                    st.error("Not enough data points (need 14+ days) for reliable ML.")
                else:
                    try:
                        model = RobustForecaster()
                        model.fit(item_data)
                        forecast = model.predict(30)
                        
                        fig = go.Figure()
                        
                        # Historical
                        fig.add_trace(go.Scatter(x=item_data['ds'], y=item_data['y'], mode='markers', name='Actual History', marker=dict(color='gray', opacity=0.5)))
                        
                        # Confidence Interval
                        fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['Upper_Bound'], mode='lines', marker=dict(color="#444"), line=dict(width=0), showlegend=False))
                        fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['Lower_Bound'], mode='lines', marker=dict(color="#444"), line=dict(width=0), fill='tonexty', fillcolor='rgba(68, 68, 68, 0.2)', name='Uncertainty Range'))
                        
                        # Prediction
                        fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['Predicted_Demand'], mode='lines+markers', name='Forecast', line=dict(color='#00CC96', width=3)))
                        
                        fig.update_layout(title=f"30-Day Forecast: {sel_item}", xaxis_title="Date", yaxis_title="Quantity")
                        st.plotly_chart(fig, use_container_width=True)
                        
                        st.markdown("#### Forecast Data Table")
                        st.dataframe(forecast[['ds', 'Predicted_Demand', 'Lower_Bound', 'Upper_Bound']], use_container_width=True)
                        
                    except Exception as e:
                        st.error(f"Modeling Error: {e}")

    # [TAB 8]
    with tab6: st.header("AI Chat"); st.info("Chat Interface Active")

else:
    st.info("👋 Please upload your data to begin.")
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
import requests # Ensure requests is imported
from itertools import combinations # For Combo AI
from collections import Counter # For Combo AI

# --- NEW IMPORTS FOR HYBRID MODEL ---
from prophet import Prophet
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

# --- PAGE CONFIG ---
st.set_page_config(page_title="Mithas Intelligence 10.5", layout="wide")

# --- DATA PROCESSING ---
@st.cache_data
def load_data(file):
    try:
        df = pd.read_excel(file)
    except Exception as e:
        st.error(f"Error reading Excel file: {e}")
        return pd.DataFrame()

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
    else:
        df['Hour'] = 0
        
    df['Hour'] = df['Hour'].fillna(0).astype(int)

    # --- FIX: REMOVE DUPLICATES TO PREVENT INFLATION ---
    df = df.drop_duplicates()
    
    return df

# --- HELPER FUNCTIONS ---

def get_pareto_items(df):
    # Filter for positive revenue only to avoid returns skewing the top list
    valid_df = df[df['TotalAmount'] > 0]
    item_rev = valid_df.groupby('ItemName')['TotalAmount'].sum().sort_values(ascending=False).reset_index()
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

# =================================================================================================
# --- NEW: AI AGENT PAYLOAD GENERATORS (STRATEGIC, CATEGORY, COMBO) ---
# =================================================================================================

def prepare_strategic_payload(df):
    """
    Prepares a comprehensive dataset for the AI Strategist (Overview Agent).
    Calculates Time-based averages, Peak Hours, and Trends.
    """
    # 1. Basic Metrics
    total_revenue = df['TotalAmount'].sum()
    total_orders = df['OrderID'].nunique()
    avg_order_value = total_revenue / total_orders if total_orders > 0 else 0

    # 2. Time-Based Averages
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Avg Revenue per Day
    daily_sales = df.groupby(df['Date'].dt.date)['TotalAmount'].sum()
    avg_rev_per_day = daily_sales.mean()

    # Avg Revenue per Week
    weekly_sales = df.groupby(df['Date'].dt.isocalendar().week)['TotalAmount'].sum()
    avg_rev_per_week = weekly_sales.mean()

    # 3. Peak Hours Data
    peak_hours = df.groupby(df['Hour'])['TotalAmount'].sum().reset_index()
    peak_hours.columns = ['Hour', 'Revenue']
    peak_hours_data = peak_hours.to_dict(orient='records')

    # 4. Revenue Trend (Daily)
    daily_trend = df.groupby(df['Date'].dt.date)['TotalAmount'].sum().reset_index()
    daily_trend.columns = ['Date', 'Revenue']
    daily_trend['Date'] = daily_trend['Date'].astype(str) # Serialize date
    trend_data = daily_trend.to_dict(orient='records')

    # 5. Weekly Performance
    weekly_trend = df.groupby(df['Date'].dt.isocalendar().week)['TotalAmount'].sum().reset_index()
    weekly_trend.columns = ['Week_Number', 'Revenue']
    weekly_performance_data = weekly_trend.to_dict(orient='records')

    payload = {
        "summary_metrics": {
            "total_revenue": round(total_revenue, 2),
            "total_orders": total_orders,
            "average_order_value": round(avg_order_value, 2),
            "average_revenue_per_day": round(avg_rev_per_day, 2),
            "average_revenue_per_week": round(avg_rev_per_week, 2)
        },
        "graph_data": {
            "peak_hours_analysis": peak_hours_data,
            "daily_revenue_trend": trend_data,
            "weekly_performance": weekly_performance_data
        }
    }
    return payload

def prepare_category_detailed_payload(df):
    """
    Prepares deep-dive dataset for Category Agent.
    Structure: Global Overview (80/20) -> Per Category -> Per Item Matrix.
    """
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df['Day_Name'] = df['Date'].dt.day_name()
    
    # 1. GLOBAL OVERVIEW (80/20 Rule)
    category_revenue = df.groupby('Category')['TotalAmount'].sum().reset_index()
    category_revenue = category_revenue.sort_values('TotalAmount', ascending=False)
    category_overview = category_revenue.to_dict(orient='records')

    # 2. DEEP DIVE DICTIONARY
    categories_data = {}
    all_categories = df['Category'].unique()
    
    for cat in all_categories:
        cat_df = df[df['Category'] == cat]
        
        # A. Category Level Trends
        cat_daily = cat_df.groupby('Day_Name')['TotalAmount'].sum().to_dict()
        cat_hourly = cat_df.groupby('Hour')['TotalAmount'].sum().to_dict()
        
        # B. Item Level Trends (The Matrix)
        items_data = []
        for item in cat_df['ItemName'].unique():
            item_df = cat_df[cat_df['ItemName'] == item]
            
            # Specific item patterns
            item_daily = item_df.groupby('Day_Name')['TotalAmount'].sum().to_dict()
            item_hourly = item_df.groupby('Hour')['TotalAmount'].sum().to_dict()
            item_total = item_df['TotalAmount'].sum()
            
            items_data.append({
                "item_name": item,
                "total_revenue": round(item_total, 2),
                "daily_pattern": item_daily,
                "hourly_pattern": item_hourly
            })
            
        categories_data[cat] = {
            "category_metrics": {
                "total_revenue": round(cat_df['TotalAmount'].sum(), 2),
                "daily_sales": cat_daily,
                "hourly_sales": cat_hourly
            },
            "items_analysis": items_data
        }

    return {
        "overview_80_20": category_overview,
        "deep_dives": categories_data
    }

def calculate_smart_combos_ai(df):
    """
    Advanced Association Rules for AI Agent: Support, Confidence, Lift, Conviction, Zhang's Metric.
    Filters out 'Gifts and Toys' and 'Display Items'.
    """
    # 1. FILTER EXCLUDED CATEGORIES
    excluded_cats = ['Gifts and Toys', 'Display Items']
    valid_df = df[~df['Category'].isin(excluded_cats)].copy()

    # 2. GROUP ORDERS
    orders = valid_df.groupby('OrderID')['ItemName'].unique().tolist()
    item_counts = valid_df['ItemName'].value_counts().to_dict()
    total_orders = len(orders)
    
    pair_counts = Counter()
    for order_items in orders:
        if len(order_items) > 1:
            pair_counts.update(combinations(sorted(order_items), 2))
            
    # 3. CALCULATE METRICS
    combos_data = []
    # Analyze top 100 pairs for efficiency in AI Payload
    for (item_a, item_b), count in pair_counts.most_common(100):
        support_ab = count / total_orders
        support_a = item_counts[item_a] / total_orders
        support_b = item_counts[item_b] / total_orders
        
        # Confidence
        conf_a_to_b = support_ab / support_a
        conf_b_to_a = support_ab / support_b
        confidence = max(conf_a_to_b, conf_b_to_a)
        
        # Lift
        lift = support_ab / (support_a * support_b) if (support_a * support_b) > 0 else 0
        
        # Conviction
        conv_a_to_b = (1 - support_b) / (1 - conf_a_to_b) if (1 - conf_a_to_b) > 0 else 100
        conv_b_to_a = (1 - support_a) / (1 - conf_b_to_a) if (1 - conf_b_to_a) > 0 else 100
        conviction = max(conv_a_to_b, conv_b_to_a)
        
        # Zhang's Metric
        numerator = support_ab - (support_a * support_b)
        if numerator >= 0:
            denominator = min(support_a * (1 - support_b), support_b * (1 - support_a))
        else:
            denominator = min(support_a * support_b, (1 - support_a) * (1 - support_b))
        
        zhangs_metric = numerator / denominator if denominator > 0 else 0

        combos_data.append({
            "items": [item_a, item_b],
            "metrics": {
                "support": round(support_ab, 4),
                "confidence": round(confidence, 2),
                "lift": round(lift, 2),
                "conviction": round(conviction, 2),
                "zhangs_metric": round(zhangs_metric, 2)
            }
        })
        
    return combos_data, valid_df

def prepare_combo_strategy_payload(df):
    """
    Combines Metrics with Schedules for the Combo Agent.
    """
    smart_combos, valid_df = calculate_smart_combos_ai(df)
    
    # Get Schedules for relevant items
    item_schedules = {}
    target_items = set()
    for c in smart_combos:
        target_items.update(c['items'])
        
    valid_df['Day'] = pd.to_datetime(valid_df['Date']).dt.day_name()
    
    for item in target_items:
        item_data = valid_df[valid_df['ItemName'] == item]
        
        # Avoid errors if item has no sales (rare if in combo, but safety first)
        if not item_data.empty:
            busy_days = item_data.groupby('Day')['TotalAmount'].sum().nlargest(3).index.tolist()
            slow_days = item_data.groupby('Day')['TotalAmount'].sum().nsmallest(3).index.tolist()
            busy_hours = item_data.groupby('Hour')['TotalAmount'].sum().nlargest(5).index.tolist()
            
            item_schedules[item] = {
                "high_revenue_days": busy_days,
                "low_revenue_days": slow_days,
                "peak_hours": busy_hours
            }

    return {
        "smart_combos_data": smart_combos,
        "item_schedules": item_schedules
    }

# =================================================================================================

# --- ADVANCED HYBRID FORECASTER (FIXED LOGIC) ---

class HybridDemandForecaster:
    def __init__(self, seasonality_mode='multiplicative'):
        # Changed daily_seasonality to False because we are feeding daily data
        self.prophet_model = Prophet(seasonality_mode=seasonality_mode, yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
        try: self.prophet_model.add_country_holidays(country_name='IN')
        except: pass 
        
        # --- FIX: REMOVED early_stopping_rounds=50 TO FIX VALIDATION DATASET ERROR ---
        self.xgb_model = XGBRegressor(n_estimators=1000, learning_rate=0.05, max_depth=6, n_jobs=-1, objective='reg:squarederror')
        
        self.rf_model = RandomForestRegressor(n_estimators=200, max_depth=10, n_jobs=-1, random_state=42)
        self.meta_model = Ridge(alpha=1.0)
        self.scaler = StandardScaler()
        self.last_training_date = None
        self.training_data_tail = None 
        self.is_fitted = False
        self.feature_columns = []

    def create_features(self, df, is_future=False):
        df_feat = df.copy()
        df_feat['dayofweek'] = df_feat['ds'].dt.dayofweek
        df_feat['quarter'] = df_feat['ds'].dt.quarter
        df_feat['month'] = df_feat['ds'].dt.month
        df_feat['is_weekend'] = df_feat['dayofweek'].apply(lambda x: 1 if x >= 5 else 0)
        
        # --- FIX: LOGIC ADAPTED FOR DAILY DATA (NOT HOURLY) ---
        # --- FIX 2: Removed "and not is_future" to ensure columns are created during prediction ---
        if 'y' in df_feat.columns:
            df_feat['lag_1d'] = df_feat['y'].shift(1)
            df_feat['lag_7d'] = df_feat['y'].shift(7)
            df_feat['rolling_mean_3d'] = df_feat['y'].rolling(window=3).mean()
            df_feat['rolling_std_7d'] = df_feat['y'].rolling(window=7).std()
            
        df_feat = df_feat.fillna(0)
        return df_feat

    def fit(self, df):
        # Ensure timezone naive
        df['ds'] = df['ds'].dt.tz_localize(None)
        
        self.last_training_date = df['ds'].max()
        self.training_data_tail = df.tail(14).copy() # Keep last 14 days for context
        self.prophet_model.fit(df)
        
        df_ml = self.create_features(df)
        drop_cols = ['ds', 'y', 'yhat']
        self.feature_columns = [c for c in df_ml.columns if c not in drop_cols and np.issubdtype(df_ml[c].dtype, np.number)]
        
        X = df_ml[self.feature_columns]
        y = df['y']
        
        # --- FIX: APPLY SCALER ---
        X_scaled = self.scaler.fit_transform(X)
        
        self.xgb_model.fit(X_scaled, y, verbose=False)
        self.rf_model.fit(X_scaled, y)
        
        pred_prophet = self.prophet_model.predict(df)['yhat'].values
        pred_xgb = self.xgb_model.predict(X_scaled)
        pred_rf = self.rf_model.predict(X_scaled)
        
        stacked_X = np.column_stack((pred_prophet, pred_xgb, pred_rf))
        self.meta_model.fit(stacked_X, y)
        self.is_fitted = True

    def predict(self, periods=30):
        if not self.is_fitted: raise Exception("Model not fitted yet.")
        
        future_prophet = self.prophet_model.make_future_dataframe(periods=periods, freq='D')
        forecast_prophet = self.prophet_model.predict(future_prophet)
        
        future_dates = future_prophet.tail(periods).copy()
        extended_df = pd.concat([self.training_data_tail, future_dates], axis=0, ignore_index=True)
        extended_feat = self.create_features(extended_df, is_future=True)
        
        X_future = extended_feat.tail(periods)[self.feature_columns]
        
        # --- FIX: TRANSFORM FUTURE DATA ---
        X_future_scaled = self.scaler.transform(X_future)
        
        pred_prophet = forecast_prophet['yhat'].tail(periods).values
        pred_xgb = self.xgb_model.predict(X_future_scaled)
        pred_rf = self.rf_model.predict(X_future_scaled)
        
        stacked_future = np.column_stack((pred_prophet, pred_xgb, pred_rf))
        final_pred = self.meta_model.predict(stacked_future)
        
        result = future_dates[['ds']].copy()
        result['Predicted_Demand'] = np.maximum(final_pred, 0)
        result['Prophet_View'] = pred_prophet
        result['XGB_View'] = pred_xgb
        result['RF_View'] = pred_rf
        return result

# --- COMBO & ASSOCIATION LOGIC (REFINED) ---

def run_advanced_association(df, level='ItemName', min_sup=0.005, min_conf=0.1, min_orders=2):
    # 1. Setup Basket
    valid_df = df[df['Quantity'] > 0]
    
    # Group by Transaction to get the "Basket"
    qty_basket = valid_df.groupby(['OrderID', level])['Quantity'].sum().unstack(fill_value=0)
    
    # Boolean Basket for FP-Growth (0/1)
    basket_sets = (qty_basket > 0).astype(bool) 
    
    # 2. Run FP-Growth
    frequent_itemsets = fpgrowth(basket_sets, min_support=min_sup, use_colnames=True)
    if frequent_itemsets.empty: return pd.DataFrame()
    
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_conf)
    
    # 3. Logic: Calculate "Times Sold Together" (Transaction Count)
    def get_transaction_stats(row):
        ant = list(row['antecedents'])[0]
        con = list(row['consequents'])[0]
        
        # Find rows where BOTH are present
        common_mask = (qty_basket[ant] > 0) & (qty_basket[con] > 0)
        orders_count = common_mask.sum()
        
        # Sum quantities in those specific common orders
        qty_ant = qty_basket.loc[common_mask, ant].sum()
        qty_con = qty_basket.loc[common_mask, con].sum()
        
        return pd.Series([orders_count, qty_ant, qty_con])

    stats = rules.apply(get_transaction_stats, axis=1)
    rules['Times Sold Together'] = stats[0].astype(int)
    rules['Qty A'] = stats[1].astype(int)
    rules['Qty B'] = stats[2].astype(int)
    
    # 4. Filter by Minimum Orders (The Noise Killer)
    rules = rules[rules['Times Sold Together'] >= min_orders]
    
    if rules.empty: return pd.DataFrame()

    # 5. Clean & Format Output
    rules['Support (%)'] = (rules['support'] * 100).round(2)
    rules['Antecedent'] = rules['antecedents'].apply(lambda x: list(x)[0])
    rules['Consequent'] = rules['consequents'].apply(lambda x: list(x)[0])
    
    # --- UPDATED: REMOVED CATEGORY SELF-REFERENCE CHECK TO FIX USER ISSUE ---
    # if level == 'Category':
    #     rules = rules[rules['Antecedent'] != rules['Consequent']]
        
    rules['Total Qty (Split)'] = rules.apply(lambda x: f"{x['Qty A'] + x['Qty B']} ({x['Qty A']} + {x['Qty B']})", axis=1)
    
    # Zhang's Metric
    def zhangs_metric(rule):
        sup = rule['support']
        sup_a = rule['antecedent support']
        sup_c = rule['consequent support']
        numerator = sup - (sup_a * sup_c)
        denominator = max(sup * (1 - sup_a), sup_a * (sup_c - sup))
        if denominator == 0: return 0
        return numerator / denominator

    rules['zhang'] = rules.apply(zhangs_metric, axis=1)
    
    # Deduplicate A->B and B->A (Keep the one with higher confidence or just first)
    rules['pair_key'] = rules.apply(lambda x: frozenset([x['Antecedent'], x['Consequent']]), axis=1)
    rules = rules.drop_duplicates(subset=['pair_key'])
    
    rules = rules.sort_values('Times Sold Together', ascending=False)
    
    # --- UPDATED: Added 'conviction' to the return list ---
    return rules[['Antecedent', 'Consequent', 'Times Sold Together', 'Support (%)', 'Total Qty (Split)', 'confidence', 'lift', 'zhang', 'conviction']]

def get_combo_analysis_full(df):
    valid_df = df[df['Quantity'] > 0]
    basket = valid_df.groupby(['OrderID', 'ItemName'])['Quantity'].count().unstack().fillna(0)
    basket_sets = (basket > 0).astype(int)
    
    frequent = apriori(basket_sets, min_support=0.005, use_colnames=True)
    if frequent.empty: return pd.DataFrame()
    
    rules = association_rules(frequent, metric="lift", min_threshold=1.05)
    
    rules['Item A'] = rules['antecedents'].apply(lambda x: list(x)[0])
    rules['Item B'] = rules['consequents'].apply(lambda x: list(x)[0])
    
    # Filter out self-matches
    rules = rules[rules['Item A'] != rules['Item B']]
    
    rules['pair'] = rules.apply(lambda x: tuple(sorted([x['Item A'], x['Item B']])), axis=1)
    rules = rules.drop_duplicates(subset='pair')
    
    item_cat_map = df.set_index('ItemName')['Category'].to_dict()
    rules['Category A'] = rules['Item A'].map(item_cat_map)
    rules['Category B'] = rules['Item B'].map(item_cat_map)
    rules['Specific Item Combo'] = rules['Item A'] + " + " + rules['Item B']
    
    total_transactions = len(basket_sets)
    rules['Times Sold Together'] = (rules['support'] * total_transactions).round().astype(int)
    
    rules['Peak Hour'] = rules.apply(lambda x: get_peak_hour_for_pair(df, x['Item A'], x['Item B']), axis=1)
    
    # Combo Value Logic
    item_avg_price = df.groupby('ItemName').apply(
        lambda x: x['TotalAmount'].sum() / x['Quantity'].sum() if x['Quantity'].sum() > 0 else 0
    ).to_dict()
    rules['Combo Value'] = rules['Item A'].map(item_avg_price).fillna(0) + rules['Item B'].map(item_avg_price).fillna(0)
    
    return rules

def get_part3_strategy(rules_df):
    if rules_df.empty: return pd.DataFrame(), pd.DataFrame()
    proven = rules_df.sort_values('Times Sold Together', ascending=False).head(10).copy()
    potential = rules_df[~rules_df.index.isin(proven.index)]
    # Added > 5 check to filter out 1-time accidental sales having high lift
    potential = potential[(potential['lift'] > 1.5) & (potential['Times Sold Together'] > 5)].sort_values('lift', ascending=False).head(10).copy()
    return proven, potential

# --- ANALYTICS MODULES (EXISTING) ---
def get_overview_metrics(df):
    total_rev = df['TotalAmount'].sum()
    
    # FIX: Calculate AOV based on valid orders (TotalAmount > 0) to avoid deflation from void bills
    valid_orders = df[df['TotalAmount'] > 0]
    total_orders = valid_orders['OrderID'].nunique()
    
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
    # Filter only positive sales for Pareto
    pos_df = df[df['TotalAmount'] > 0]
    item_rev = pos_df.groupby(['Category', 'ItemName'])['TotalAmount'].sum().reset_index()
    total_revenue = item_rev['TotalAmount'].sum()
    item_rev = item_rev.sort_values('TotalAmount', ascending=False)
    item_rev['Cumulative'] = item_rev['TotalAmount'].cumsum()
    item_rev['CumPerc'] = 100 * item_rev['Cumulative'] / total_revenue
    pareto_items = item_rev[item_rev['CumPerc'] <= 80].copy()
    total_unique_items = pos_df['ItemName'].nunique()
    pareto_unique_items = pareto_items['ItemName'].nunique()
    ratio_text = f"**{pareto_unique_items} items** (out of {total_unique_items}) contribute to 80% of your revenue."
    percentage_of_menu = (pareto_unique_items / total_unique_items) * 100
    cat_rev = pos_df.groupby('Category')['TotalAmount'].sum().reset_index()
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

# --- MAIN APP LAYOUT ---
st.title("📊 Mithas Restaurant Intelligence 10.5")
uploaded_file = st.sidebar.file_uploader("Upload Monthly Data (Sidebar)", type=['xlsx'])

# --- NEW: SIDEBAR AI CONTROLS ---
st.sidebar.divider()
st.sidebar.subheader("🤖 AI Reporting Agent")
n8n_webhook = st.sidebar.text_input("n8n Webhook URL", placeholder="https://your-n8n.com/webhook/...")
# Trigger logic moved to Tab 9

if uploaded_file:
    df = load_data(uploaded_file)
    pareto_list = get_pareto_items(df)
    pareto_count = len(pareto_list)
    
    rules_df = get_combo_analysis_full(df)
    proven_df, potential_df = get_part3_strategy(rules_df)
    
    proven_list = []
    potential_list = []
    if not proven_df.empty: proven_list = proven_df['pair'].tolist()
    if not potential_df.empty: potential_list = potential_df['pair'].tolist()
    
    # --- UPDATED TABS LIST with 'AI Reports' ---
    tab1, tab_day_wise, tab_cat, tab2, tab3, tab4, tab_assoc, tab5, tab6, tab_ai = st.tabs([
        "Overview", "Day wise Analysis", "Category Details", "Pareto (Visual)", "Time Series", "Smart Combos", "Association Analysis", "Demand Forecast", "AI Chat", "🤖 AI Reports"
    ])

    # --- TAB 1: OVERVIEW ---
    with tab1:
        st.header("🏢 Business Overview")
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
                
                # --- UPDATED: REMOVED BOLD (Arial Black) AND EXPLICIT BLACK COLOR ---
                fig_daily.update_xaxes(
                    tickfont=dict(size=14), # Removed family='Arial Black' and color='black'
                    title_font=dict(size=16)
                )
                st.plotly_chart(fig_daily, use_container_width=True)
            
            # --- NEW ADDITION: WEEKLY SALES GRAPH ---
            st.subheader("📅 Weekly Sales Performance")
            if 'Date' in df.columns:
                weekly_df = df.copy()
                # Determine the start of the week (Monday) for each date
                weekly_df['WeekStart'] = weekly_df['Date'] - pd.to_timedelta(weekly_df['Date'].dt.dayofweek, unit='D')
                weekly_df['WeekEnd'] = weekly_df['WeekStart'] + pd.to_timedelta(6, unit='D')
                
                # Aggregate sales by week
                weekly_stats = weekly_df.groupby(['WeekStart', 'WeekEnd'])['TotalAmount'].sum().reset_index()
                weekly_stats = weekly_stats.sort_values('WeekStart')
                
                # Create labels for the x-axis
                weekly_stats['Week Label'] = [
                    f"Week {i+1} ({row['WeekStart'].strftime('%d %b')} - {row['WeekEnd'].strftime('%d %b')})"
                    for i, row in weekly_stats.iterrows()
                ]
                
                fig_weekly = px.bar(weekly_stats, x='Week Label', y='TotalAmount')
                fig_weekly.update_xaxes(title=None)
                st.plotly_chart(fig_weekly, use_container_width=True)
            
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
        
        st.subheader("⚠️ Items Not Worth Producing")
        st.markdown("Items with **< 3 units sold** in a 3-Day window. **Refined:** Only items with >0 sales on the specific day are shown.")
        
        days_list = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
        
        available_categories = sorted(df['Category'].unique().tolist())
        dropdown_options = ["All Categories"] + available_categories
        
        selected_nw_category = st.selectbox(
            "Filter by Category",
            options=dropdown_options,
            index=0, 
            key="nw_category_dropdown"
        )

        not_worth_dict = {}
        max_len = 0
        
        for d in days_list:
            curr_idx = days_list.index(d)
            prev_d = days_list[(curr_idx - 1) % 7]
            next_d = days_list[(curr_idx + 1) % 7]
            window_days = [prev_d, d, next_d]
            
            window_df = df[df['DayOfWeek'].isin(window_days)]
            
            if selected_nw_category != "All Categories":
                window_df = window_df[window_df['Category'] == selected_nw_category]
            
            item_sums = window_df.groupby('ItemName')['Quantity'].sum()
            candidate_items = item_sums[item_sums < 3].index.tolist()
            
            day_specific_df = df[df['DayOfWeek'] == d]
            if selected_nw_category != "All Categories":
                day_specific_df = day_specific_df[day_specific_df['Category'] == selected_nw_category]
            
            active_items_today = day_specific_df[day_specific_df['Quantity'] > 0]['ItemName'].unique().tolist()
            final_items = [item for item in candidate_items if item in active_items_today]
            final_items = [f"★ {x}" if x in pareto_list else x for x in final_items]

            not_worth_dict[d] = final_items
            if len(final_items) > max_len:
                max_len = len(final_items)
        
        for d in days_list:
            current_len = len(not_worth_dict[d])
            if current_len < max_len:
                not_worth_dict[d].extend([""] * (max_len - current_len))
        
        waste_df = pd.DataFrame(not_worth_dict)
        st.dataframe(waste_df, use_container_width=True, height=500)

        st.markdown("---")
        st.subheader("📉 Missed Upsell Opportunities (Orders Below AOV)")
        
        # Fixed robust mean
        monthly_aov = df[df['TotalAmount'] > 0]['TotalAmount'].mean()
        
        daily_aov_stats = df[df['TotalAmount'] > 0].groupby(df['Date'].dt.date).apply(
            lambda x: pd.Series({
                'Total Orders': len(x),
                'Below AOV': (x['TotalAmount'] < monthly_aov).sum(),
                'Percentage Below AOV': ((x['TotalAmount'] < monthly_aov).sum() / len(x)) * 100
            })
        ).reset_index()
        daily_aov_stats.rename(columns={'Date': 'Day'}, inplace=True)
        
        fig_upsell = px.bar(
            daily_aov_stats,
            x='Day',
            y='Percentage Below AOV',
            title=f"Daily % of Orders Below Monthly Average (₹{monthly_aov:,.0f})",
            color='Percentage Below AOV',
            color_continuous_scale='RdYlGn_r', 
            labels={'Percentage Below AOV': '% Low Value Orders'},
            hover_data=['Total Orders', 'Below AOV']
        )
        
        fig_upsell.add_hline(
            y=50, 
            line_dash="dot", 
            line_color="red", 
            annotation_text="Critical Threshold (50%)", 
            annotation_position="top right"
        )
        
        fig_upsell.update_layout(
            xaxis_title=None,
            yaxis_title="% of Orders < AOV",
            coloraxis_showscale=False,
            yaxis_range=[0, 100],
            hovermode="x unified"
        )
        
        st.plotly_chart(fig_upsell, use_container_width=True)
        st.caption("Red bars indicate days where a high percentage of customers bought less than the average amount. These days require sales training.")

    # --- TAB: CATEGORY DETAILS ---
    with tab_cat:
        st.header("📂 Category Deep-Dive")
        cats = sorted(df['Category'].unique()) 
        total_business_rev = df['TotalAmount'].sum()
        days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        
        selected_cat_deep_dive = st.selectbox("Select Category to Analyze", cats, key='cat_deep_dive_select')
        
        cat_data = df[df['Category'] == selected_cat_deep_dive]
        
        # --- NEW: Added Revenue and Contribution stats to the header ---
        cat_rev = cat_data['TotalAmount'].sum()
        cat_perc = (cat_rev / total_business_rev) * 100
        st.subheader(f"🔹 {selected_cat_deep_dive} (Revenue: ₹{cat_rev:,.0f} | Contribution: {cat_perc:.2f}%)")
        
        # --- NEW ADDITION: CATEGORY SUMMARY STATS ---
        cat_stats_temp = cat_data.groupby('ItemName').agg({'TotalAmount': 'sum', 'Quantity': 'sum'}).reset_index()
        total_items_count = len(cat_stats_temp)
        
        # Calculate Critical Items (**)
        crit_df = cat_stats_temp[cat_stats_temp['TotalAmount'] < 500]
        crit_count = len(crit_df)
        crit_qty = crit_df['Quantity'].sum()
        
        # Calculate Warning Items (*)
        warn_df = cat_stats_temp[(cat_stats_temp['TotalAmount'] >= 500) & (cat_stats_temp['TotalAmount'] < 1500)]
        warn_count = len(warn_df)
        warn_qty = warn_df['Quantity'].sum()
        
        # Calculate Star Items (★) - Pareto
        star_df = cat_stats_temp[cat_stats_temp['ItemName'].isin(pareto_list)]
        star_count = len(star_df)
        star_qty = star_df['Quantity'].sum()
        
        # Display Summary
        st.markdown(f"**Total Items:** {total_items_count} | **`**` Items:** {crit_count} (Qty: {int(crit_qty)}) | **`*` Items:** {warn_count} (Qty: {int(warn_qty)}) | **`★` Items:** {star_count} (Qty: {int(star_qty)})")
        # --------------------------------------------
        
        cat_stats = cat_data.groupby('ItemName').agg({'TotalAmount': 'sum', 'Quantity': 'sum'}).reset_index()
        cat_stats['Contribution %'] = (cat_stats['TotalAmount'] / total_business_rev) * 100
        
        day_pivot = cat_data.groupby(['ItemName', 'DayOfWeek'])['Quantity'].sum().unstack(fill_value=0)
        for day in days_order:
            if day not in day_pivot.columns: day_pivot[day] = 0
        day_pivot = day_pivot[days_order] 
        
        cat_stats = pd.merge(cat_stats, day_pivot, on='ItemName', how='left').fillna(0)
        
        # --- FIX: Force Quantity to be integer ---
        cat_stats['Quantity'] = cat_stats['Quantity'].astype(int)
        
        cat_stats = cat_stats.sort_values('TotalAmount', ascending=False)
        
        def mark_item_name(row):
            name = row['ItemName']
            rev = row['TotalAmount']
            if name in pareto_list:
                name = f"★ {name}"
            if rev < 500:
                name = f"{name} **"
            elif 500 <= rev < 1500:
                name = f"{name} *"
            return name

        cat_stats['Item Name'] = cat_stats.apply(mark_item_name, axis=1)
        
        col_config = {
            "TotalAmount": st.column_config.NumberColumn("Revenue", format="₹%d"),
            "Contribution %": st.column_config.ProgressColumn("Contribution", format="%.2f%%"),
            # --- FIX: Added format="%d" to ensure clean integer display for Total Qty ---
            "Quantity": st.column_config.NumberColumn("Total Qty", format="%d")
        }
        for day in days_order:
            # --- MODIFIED: Calculate total for the specific day and add to header ---
            day_sum = day_pivot[day].sum()
            col_config[day] = st.column_config.NumberColumn(f"{day} ({int(day_sum)})", format="%d")
        
        cols_to_show = ['Item Name', 'TotalAmount', 'Quantity', 'Contribution %'] + days_order
        
        # --- FIX: Apply row-wise heatmap for the 7 days ---
        cat_styler = cat_stats[cols_to_show].style.background_gradient(subset=days_order, axis=1, cmap="YlOrRd")
        st.dataframe(cat_styler, column_config=col_config, hide_index=True, use_container_width=True)
        
        st.caption("📝 **Legend:** `**` = Revenue < ₹500 (Critical) | `*` = Revenue < ₹1500 (Warning) | `★` = Pareto Top 80% Item")
        st.divider()

        st.subheader(f"Hourly Breakdown Matrix ({selected_cat_deep_dive})")
        st.markdown("Drill down: **Day** → **Date** → **Hourly Matrix**.")
        
        day_selected_mat = st.selectbox("1. Select Day of Week", days_order, key='mat_day_select')
        
        day_df_mat = df[df['DayOfWeek'] == day_selected_mat]
        
        if not day_df_mat.empty:
            unique_dates_mat = sorted(day_df_mat['Date'].dt.strftime('%Y-%m-%d').unique())
            date_options_mat = [f"All {day_selected_mat}s Combined"] + unique_dates_mat
            date_selected_mat = st.selectbox(f"2. Select Date ({day_selected_mat})", date_options_mat, key='mat_date_select')
            
            if date_selected_mat == f"All {day_selected_mat}s Combined":
                target_df_mat = day_df_mat
            else:
                target_df_mat = day_df_mat[day_df_mat['Date'].dt.strftime('%Y-%m-%d') == date_selected_mat]
            
            target_df_mat = target_df_mat[target_df_mat['Category'] == selected_cat_deep_dive]
            
            if not target_df_mat.empty:
                df_3day_mat = pd.DataFrame()
                if date_selected_mat == f"All {day_selected_mat}s Combined":
                    curr_idx = days_order.index(day_selected_mat)
                    prev_day = days_order[(curr_idx - 1) % 7]
                    next_day = days_order[(curr_idx + 1) % 7]
                    target_days = [prev_day, day_selected_mat, next_day]
                    df_3day_mat = df[df['DayOfWeek'].isin(target_days)]
                else:
                    sel_dt_mat = pd.to_datetime(date_selected_mat)
                    target_dates_mat = [sel_dt_mat - timedelta(days=1), sel_dt_mat, sel_dt_mat + timedelta(days=1)]
                    df_3day_mat = df[df['Date'].isin(target_dates_mat)]

                df_3day_mat = df_3day_mat[df_3day_mat['Category'] == selected_cat_deep_dive]
                qty_3day_mat = df_3day_mat.groupby('ItemName')['Quantity'].sum().rename("3-Day Qty")

                pivot_mat = target_df_mat.groupby(['ItemName', 'Hour'])['Quantity'].sum().unstack(fill_value=0)
                for h in range(9, 24):
                    if h not in pivot_mat.columns: pivot_mat[h] = 0
                pivot_mat = pivot_mat[sorted(pivot_mat.columns)]
                pivot_mat.columns = [f"{int(h)}-{int(h)+1}" for h in pivot_mat.columns]
                pivot_mat['Total Quantity'] = pivot_mat.sum(axis=1)
                pivot_mat = pivot_mat.join(qty_3day_mat, how='left').fillna(0)
                pivot_mat['3-Day Qty'] = pivot_mat['3-Day Qty'].astype(int)
                pivot_mat = pivot_mat.sort_values('Total Quantity', ascending=False)
                pivot_mat.index = pivot_mat.index.map(lambda x: f"★ {x}" if x in pareto_list else x)
                
                # --- FIX: Ensure integers to avoid "22.000000" display issues ---
                pivot_mat = pivot_mat.astype(int)
                
                # --- FIX: Apply row-wise heatmap for the hourly columns only ---
                hour_cols = [c for c in pivot_mat.columns if c not in ['Total Quantity', '3-Day Qty']]
                
                # Added .format() to ensure clean number display
                mat_styler = pivot_mat.style.format("{:.0f}").background_gradient(subset=hour_cols, axis=1, cmap="YlOrRd")
                st.dataframe(mat_styler, use_container_width=True, height=600)
            else:
                st.warning(f"No sales found for category '{selected_cat_deep_dive}' on this selection.")
        else:
            st.warning(f"No transactions found for {day_selected_mat} in the uploaded file.")
        
        st.divider()

    # --- TAB 2: PARETO ---
    with tab2:
        st.header("🏆 Pareto Analysis")
        pareto_df, ratio_msg, menu_perc = analyze_pareto_hierarchical(df)
        st.info(f"💡 **Insight:** {ratio_msg} (Only {menu_perc:.1f}% of your menu!)")
        pareto_df['ItemName'] = pareto_df['ItemName'].apply(lambda x: f"★ {x}")
        st.dataframe(pareto_df, column_config={"CatContrib": st.column_config.NumberColumn("Category Share %", format="%.2f%%"), "ItemContrib": st.column_config.NumberColumn("Item Share % (Global)", format="%.2f%%"), "TotalAmount": st.column_config.NumberColumn("Revenue", format="₹%d")}, hide_index=True, height=600, use_container_width=True)

    # --- TAB 3: TIME SERIES ---
    with tab3:
        st.header("📅 Daily Trends")
        n_ts = st.slider("Number of items per category (Top N by Revenue)", 5, 30, 5)
        plot_time_series_fixed(df, pareto_list, n_ts)

    # --- TAB 4: SMART COMBOS ---
    with tab4:
        st.header("🍔 Smart Combo Strategy")
        with st.expander("🛠️ Reorder Combo Layout"):
            combo_order = st.multiselect("Section Order", ["Part 1: Full Combo Map", "Part 3: Strategic Recommendations"], default=["Part 1: Full Combo Map", "Part 3: Strategic Recommendations"])
        
        def star_combo_string(item_name):
            if item_name in pareto_list:
                return f"★ {item_name}"
            return item_name

        def process_combo_df_for_display(input_df):
            if input_df.empty: return input_df
            df_display = input_df.copy()
            if 'Item A' in df_display.columns and 'Item B' in df_display.columns:
                df_display['Item A Display'] = df_display['Item A'].apply(star_combo_string)
                df_display['Item B Display'] = df_display['Item B'].apply(star_combo_string)
                df_display['Specific Item Combo'] = df_display['Item A Display'] + " + " + df_display['Item B Display']
            return df_display

        rules_display = process_combo_df_for_display(rules_df)
        proven_display = process_combo_df_for_display(proven_df)
        potential_display = process_combo_df_for_display(potential_df)

        def render_part1():
            st.subheader("1️⃣ Part 1: Full Category + Item Combo Map")
            if not rules_display.empty:
                display_cols = ['Category A', 'Category B', 'Specific Item Combo', 'Times Sold Together', 'Combo Value', 'Peak Hour', 'lift']
                st.dataframe(rules_display[display_cols].sort_values('Times Sold Together', ascending=False), column_config={"Specific Item Combo": st.column_config.TextColumn("Item A + Item B", width="medium"), "lift": st.column_config.NumberColumn("Lift Strength", format="%.2f"), "Combo Value": st.column_config.NumberColumn("Combo Value", format="₹%.2f")}, hide_index=True, use_container_width=True)
            else: st.warning("No significant combos found.")
        def render_part3():
            st.subheader("3️⃣ Part 3: Strategic Recommendations")
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("#### 🔥 Proven Winners")
                if not proven_display.empty: st.dataframe(proven_display[['Specific Item Combo', 'Times Sold Together', 'Combo Value', 'Peak Hour']], column_config={"Combo Value": st.column_config.NumberColumn("Combo Value", format="₹%.2f")}, hide_index=True, use_container_width=True)
                else: st.info("No data.")
            with c2:
                st.markdown("#### 💎 Hidden Gems")
                if not potential_display.empty: st.dataframe(potential_display[['Specific Item Combo', 'lift', 'Combo Value', 'Peak Hour']], column_config={"lift": st.column_config.NumberColumn("Compatibility Score", format="%.2f"), "Combo Value": st.column_config.NumberColumn("Combo Value", format="₹%.2f")}, hide_index=True, use_container_width=True)
                else: st.info("No hidden gems found.")
        combo_map = {"Part 1: Full Combo Map": render_part1, "Part 3: Strategic Recommendations": render_part3}
        for block in combo_order:
            if block in combo_map: combo_map[block]()
    
    # --- TAB 5: ASSOCIATION ANALYSIS (OPTIMIZED) ---
    with tab_assoc:
        st.header("🧬 Scientific Association Analysis")
        st.markdown("Discover hidden patterns. **Note:** Rules are filtered to exclude 'One-Off' party orders.")
        
        c1, c2, c3 = st.columns(3)
        with c1:
            analysis_level = st.radio("Analysis Level", ["ItemName", "Category"], horizontal=True)
        
        # --- PRE-CALCULATIONS FOR SLIDERS ---
        valid_df_global = df[df['Quantity'] > 0]
        basket_global = valid_df_global.groupby(['OrderID', analysis_level])['Quantity'].count().unstack().fillna(0)
        basket_sets_global = (basket_global > 0).astype(bool)
        
        # Smart Defaults
        supports = basket_sets_global.mean().sort_values(ascending=False)
        max_sup = supports.max() if not supports.empty else 1.0
        
        max_sup_percent = float(min(100.0, max_sup * 100 + 1.0)) 
        
        with c2:
            # Min Support Slider
            min_support_percent = st.slider("Minimum Support (%)", 0.01, max_sup_percent, 0.1, step=0.01, help="Percentage of total transactions the combo must appear in.")
            min_support_val = min_support_percent / 100.0
            
        with c3:
            # NEW: Min Transactions Slider (The Noise Filter)
            min_orders = st.slider("Min. Orders Together", 2, 50, 3, help="Remove rules that happened fewer than X times. Essential to remove one-off catering orders.")

        st.divider()
        
        with st.spinner("Running Advanced Association Algorithm..."):
            # Pass min_orders to the function
            assoc_rules = run_advanced_association(df, level=analysis_level, min_sup=min_support_val, min_orders=min_orders)
            
            if not assoc_rules.empty:
                # Add Status logic
                def get_status(row):
                    if analysis_level != 'ItemName': return "Category Level"
                    current_pair = tuple(sorted([row['Antecedent'], row['Consequent']]))
                    if current_pair in proven_list: return "🔥 Proven Winner"
                    elif current_pair in potential_list: return "💎 Hidden Gem"
                    return "Normal"

                assoc_rules['Status'] = assoc_rules.apply(get_status, axis=1)
                
                # Apply Stars
                if analysis_level == 'ItemName':
                    assoc_rules['Antecedent'] = assoc_rules['Antecedent'].apply(lambda x: f"★ {x}" if x in pareto_list else x)
                    assoc_rules['Consequent'] = assoc_rules['Consequent'].apply(lambda x: f"★ {x}" if x in pareto_list else x)

                # DISPLAY
                # --- UPDATED: Added 'conviction' to column config ---
                st.dataframe(
                    assoc_rules,
                    column_config={
                        "Status": st.column_config.TextColumn("Strategic Status"),
                        "Times Sold Together": st.column_config.NumberColumn("Orders Count", format="%d 🛒"),
                        "Support (%)": st.column_config.NumberColumn("Support", format="%.2f%%"),
                        "Total Qty (Split)": st.column_config.TextColumn("Total Qty (A + B)", width="medium"),
                        "confidence": st.column_config.NumberColumn("Confidence", format="%.2f"),
                        "lift": st.column_config.NumberColumn("Lift", format="%.2f"),
                        "zhang": st.column_config.NumberColumn("Zhang's Metric", format="%.2f"),
                        "conviction": st.column_config.NumberColumn("Conviction", format="%.2f")
                    },
                    hide_index=True, 
                    use_container_width=True, 
                    height=600
                )
                
                # Plot
                fig = px.scatter(
                    assoc_rules, x="Support (%)", y="confidence", 
                    size="Times Sold Together", color="lift",
                    hover_data=["Antecedent", "Consequent", "Total Qty (Split)"],
                    title=f"Association Rules Landscape ({analysis_level} Level)",
                    color_continuous_scale=px.colors.diverging.RdBu
                )
                st.plotly_chart(fig, use_container_width=True)
            
            else:
                st.warning(f"No rules found with >{min_orders} orders together. Try lowering the filters.")

    # --- TAB 7: DEMAND FORECAST ---
    with tab5:
        st.header("🔮 Demand Prediction (Ensemble AI)")
        st.markdown("**Model:** Hybrid Ensemble (Prophet + XGBoost + Random Forest + Meta-Learner).")
        st.markdown("---")
        
        forecast_file = st.file_uploader("Upload Historical Master File (Recommended: 3+ Months Data)", type=['xlsx'], key='forecast_uploader')
        
        df_to_use = None
        if forecast_file:
            df_to_use = load_data(forecast_file)
            st.success("✅ Using Master History File for Training")
        elif df is not None:
            df_to_use = df
            st.warning("⚠️ Using Sidebar File (Short-term data). Accuracy may be low without history.")
        
        if df_to_use is not None:
            all_items = df_to_use['ItemName'].unique()
            selected_item = st.selectbox("Select Item to Forecast", all_items)
            
            if st.button("Generate AI Forecast"):
                with st.spinner(f"Training AI Models for {selected_item}..."):
                    item_df = df_to_use[df_to_use['ItemName'] == selected_item].groupby('Date')['Quantity'].sum().reset_index()
                    item_df.columns = ['ds', 'y']
                    
                    if len(item_df) < 14:
                        st.error(f"⚠️ Not enough history for {selected_item} (Need 14+ days). Model cannot train.")
                    else:
                        try:
                            forecaster = HybridDemandForecaster()
                            forecaster.fit(item_df)
                            forecast = forecaster.predict(periods=30)
                            
                            st.subheader(f"Forecast for {selected_item}")
                            
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(x=item_df['ds'], y=item_df['y'], mode='markers', name='Actual Sales', marker=dict(color='gray', opacity=0.5, size=8)))
                            fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['Predicted_Demand'], mode='lines+markers', name='Ensemble Prediction', line=dict(color='#00CC96', width=3)))
                            fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['Prophet_View'], mode='lines', name='Prophet View', line=dict(dash='dot', color='blue', width=1), visible='legendonly'))
                            fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['XGB_View'], mode='lines', name='XGBoost View', line=dict(dash='dot', color='red', width=1), visible='legendonly'))
                            
                            fig.update_layout(
                                title="30-Day Demand Forecast", 
                                xaxis_title="Date", 
                                yaxis_title="Predicted Quantity", 
                                template="plotly_white"
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            
                            st.markdown("#### Detailed Forecast Data")
                            st.dataframe(forecast[['ds', 'Predicted_Demand', 'Prophet_View', 'XGB_View']], hide_index=True, use_container_width=True)
                            
                        except Exception as e:
                            st.error(f"Modeling Error: {e}")
        else:
            st.info("Please upload data to begin forecasting.")

    # --- TAB 8: AI CHAT ---
    with tab6:
        st.subheader("🤖 Manager Chat")
        if "messages" not in st.session_state: st.session_state.messages = []
        for msg in st.session_state.messages: st.chat_message(msg["role"]).write(msg["content"])
        if prompt := st.chat_input("Ask about combos..."):
            st.chat_message("user").write(prompt)
            st.session_state.messages.append({"role": "user", "content": prompt})
            try:
                llm = ChatOpenAI(model="gpt-4o-mini", api_key=st.secrets["OPENAI_API_KEY"])
                response = llm.invoke([SystemMessage(content="Restaurant Analyst"), HumanMessage(content=prompt)])
                st.chat_message("assistant").write(response.content)
                st.session_state.messages.append({"role": "assistant", "content": response.content})
            except: st.error("Check API Key")

    # --- TAB 9: AI AGENT REPORTS (NEW COMMAND CENTER) ---
    with tab_ai:
        st.subheader("🤖 AI Agent Command Center")
        st.info("Trigger autonomous AI agents via n8n to analyze your store data and generate comprehensive reports.")
        
        if not webhook_url:
            st.warning("⚠️ Please enter your n8n Webhook URL in the sidebar settings to use these features.")
        else:
            ai_col1, ai_col2, ai_col3 = st.columns(3)
            
            # Button 1: Strategic Overview
            with ai_col1:
                st.markdown("#### 📝 Strategic Overview")
                st.caption("Analyzes Revenue, Peak Hours, and Monthly Trends.")
                if st.button("Generate Strategy Report"):
                    with st.spinner("Packaging data & sending to AI Strategist..."):
                        try:
                            payload = prepare_strategic_payload(df)
                            response = requests.post(webhook_url, json=payload)
                            if response.status_code == 200:
                                st.success("✅ Request Sent! Check your Google Drive/Email.")
                            else:
                                st.error(f"❌ Error {response.status_code}: {response.text}")
                        except Exception as e:
                            st.error(f"Failed: {e}")

            # Button 2: Detailed Category Deep Dive
            with ai_col2:
                st.markdown("#### 🔍 Category Deep-Dive")
                st.caption("Analyzes 80/20 Rule and Item-level Hourly Matrices.")
                if st.button("Generate Category Report"):
                    with st.spinner("Calculating Item Matrices..."):
                        try:
                            payload = prepare_category_detailed_payload(df)
                            response = requests.post(webhook_url, json=payload)
                            if response.status_code == 200:
                                st.success("✅ Request Sent! Check your Google Drive.")
                            else:
                                st.error(f"❌ Error {response.status_code}: {response.text}")
                        except Exception as e:
                            st.error(f"Failed: {e}")

            # Button 3: Advanced Combo Strategy
            with ai_col3:
                st.markdown("#### 🍔 Advanced Combo Strategy")
                st.caption("Cross-references Lift, Zhang's Metric, and Item Schedules.")
                if st.button("Generate Combo Strategy"):
                    with st.spinner("Mining Associations & Schedules..."):
                        try:
                            payload = prepare_combo_strategy_payload(df)
                            response = requests.post(webhook_url, json=payload)
                            if response.status_code == 200:
                                st.success("✅ Request Sent! Check your Google Drive.")
                            else:
                                st.error(f"❌ Error {response.status_code}: {response.text}")
                        except Exception as e:
                            st.error(f"Failed: {e}")

else:
    st.info("👋 Upload data to begin.")
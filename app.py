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

# --- PAGE CONFIG ---
st.set_page_config(page_title="Mithas Analytics Pro", layout="wide", initial_sidebar_state="expanded")

# --- DATA PROCESSING ---
@st.cache_data
def load_data(file):
    df = pd.read_excel(file)
    
    # Standardize Columns
    col_map = {
        'Invoice No.': 'OrderID', 'Item Name': 'ItemName', 'Qty.': 'Quantity',
        'Final Total': 'TotalAmount', 'Price': 'UnitPrice', 'Category': 'Category',
        'Timestamp': 'Time', 'Date': 'Date'
    }
    df = df.rename(columns=col_map)
    
    # Numeric Cleanup
    for c in ['Quantity', 'TotalAmount', 'UnitPrice']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
    
    # Date Cleanup
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df['DayOfWeek'] = df['Date'].dt.day_name()
    
    # Hour Extraction
    if 'Time' in df.columns:
        # Try-catch for time parsing
        try:
            df['Hour'] = pd.to_datetime(df['Time'].astype(str), format='%H:%M:%S', errors='coerce').dt.hour
            if df['Hour'].isnull().all():
                 df['Hour'] = pd.to_datetime(df['Time'], errors='coerce').dt.hour
        except:
            df['Hour'] = 0
            
    return df

# --- ANALYTICS MODULES ---

def get_overview_metrics(df):
    """Calculates Tab 1 KPI Cards"""
    total_rev = df['TotalAmount'].sum()
    total_orders = df['OrderID'].nunique()
    
    # Time-based averages
    num_days = df['Date'].nunique()
    avg_rev_day = total_rev / num_days if num_days > 0 else 0
    
    # Weekly Average (Approximate if data < 1 week)
    num_weeks = max(1, num_days / 7)
    avg_rev_week = total_rev / num_weeks
    
    aov = total_rev / total_orders if total_orders > 0 else 0
    
    return total_rev, total_orders, avg_rev_day, avg_rev_week, aov

def analyze_peak_days(df):
    """Sorts days by total revenue"""
    days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    daily = df.groupby('DayOfWeek')['TotalAmount'].sum().reindex(days_order).reset_index()
    return daily

def analyze_peak_hour_items(df):
    """Finds top 3 peak hours and lists items sold then"""
    if 'Hour' not in df.columns: return pd.DataFrame(), []
    
    hourly_rev = df.groupby('Hour')['TotalAmount'].sum()
    # Get top 3 hours
    top_3_hours = hourly_rev.nlargest(3).index.tolist()
    
    # Filter data for those hours
    peak_df = df[df['Hour'].isin(top_3_hours)]
    top_items = peak_df.groupby('ItemName')['Quantity'].sum().sort_values(ascending=False).head(10).reset_index()
    
    return top_items, top_3_hours

def get_contribution_lists(df):
    """Calculates % contribution for Categories and Items"""
    total_rev = df['TotalAmount'].sum()
    
    # Category Level
    cat_df = df.groupby('Category')['TotalAmount'].sum().reset_index()
    cat_df['Contribution %'] = (cat_df['TotalAmount'] / total_rev) * 100
    cat_df = cat_df.sort_values('TotalAmount', ascending=False)
    
    # Item Level (within Category)
    item_df = df.groupby(['Category', 'ItemName'])['TotalAmount'].sum().reset_index()
    item_df['Contribution %'] = (item_df['TotalAmount'] / total_rev) * 100
    item_df = item_df.sort_values('TotalAmount', ascending=False)
    
    # Star Items (Global Top 20)
    stars = df.groupby('ItemName')['TotalAmount'].sum().reset_index()
    stars['Contribution %'] = (stars['TotalAmount'] / total_rev) * 100
    stars = stars.sort_values('TotalAmount', ascending=False).head(20)
    
    return cat_df, item_df, stars

def analyze_pareto(df):
    """Pareto Logic for Tab 2"""
    item_rev = df.groupby(['ItemName', 'Category'])['TotalAmount'].sum().reset_index()
    item_rev = item_rev.sort_values('TotalAmount', ascending=False)
    
    total_revenue = item_rev['TotalAmount'].sum()
    item_rev['Cumulative'] = item_rev['TotalAmount'].cumsum()
    item_rev['CumPerc'] = 100 * item_rev['Cumulative'] / total_revenue
    
    top_80 = item_rev[item_rev['CumPerc'] <= 82]
    return top_80

def get_basket_rules(df, group_col='ItemName', min_conf=0.6):
    basket = (df.groupby(['OrderID', group_col])['Quantity']
              .sum().unstack().reset_index().fillna(0)
              .set_index('OrderID'))
    basket_sets = basket.applymap(lambda x: 1 if x > 0 else 0)
    frequent = apriori(basket_sets, min_support=0.005, use_colnames=True)
    if frequent.empty: return pd.DataFrame()
    rules = association_rules(frequent, metric="confidence", min_threshold=min_conf)
    rules['antecedents'] = rules['antecedents'].apply(lambda x: list(x)[0])
    rules['consequents'] = rules['consequents'].apply(lambda x: list(x)[0])
    return rules[['antecedents', 'consequents', 'confidence', 'lift']].sort_values('confidence', ascending=False)

def plot_time_series_top5(df):
    categories = df['Category'].unique()
    for cat in categories:
        st.subheader(f"📈 {cat}: Top 5 Items Trend")
        cat_data = df[df['Category'] == cat]
        top_items = cat_data.groupby('ItemName')['Quantity'].sum().nlargest(5).index.tolist()
        subset = cat_data[cat_data['ItemName'].isin(top_items)]
        daily = subset.groupby(['Date', 'ItemName'])['Quantity'].sum().reset_index()
        if daily.empty: continue
        fig = px.line(daily, x='Date', y='Quantity', color='ItemName', markers=True)
        for item in top_items:
            avg_val = daily[daily['ItemName'] == item]['Quantity'].mean()
            fig.add_hline(y=avg_val, line_dash="dot", annotation_text=f"Avg {item}", annotation_position="top left")
        st.plotly_chart(fig, use_container_width=True)

def correlation_heatmap(df, group_col):
    pivot = df.pivot_table(index='Date', columns=group_col, values='Quantity', aggfunc='sum').fillna(0)
    corr = pivot.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=False, cmap='coolwarm', ax=ax)
    return fig

def forecast_demand(df):
    daily = df.groupby(['Date', 'ItemName'])['Quantity'].sum().reset_index()
    forecasts = []
    items = daily['ItemName'].unique()
    last_date = daily['Date'].max()
    next_30_days = [last_date + timedelta(days=x) for x in range(1, 31)]
    for item in items:
        item_data = daily[daily['ItemName'] == item].sort_values('Date')
        avg_qty = item_data.tail(7)['Quantity'].mean() if len(item_data) >= 7 else item_data['Quantity'].mean()
        for date in next_30_days:
            forecasts.append({'Date': date.strftime('%Y-%m-%d'), 'ItemName': item, 'Predicted_Qty': round(avg_qty, 1)})
    return pd.DataFrame(forecasts)

# --- MAIN APP LAYOUT ---
st.title("📊 Mithas Restaurant Intelligence 3.0")
uploaded_file = st.sidebar.file_uploader("Upload Monthly/Daily Excel", type=['xlsx'])

if uploaded_file:
    df = load_data(uploaded_file)
    if df['Date'].isnull().all():
        st.error("⚠️ Error: 'Date' column is missing or empty.")
        st.stop()

    # TABS
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Overview", "Pareto Analysis", "Time Series", "Basket & Correlations", "Forecast", "AI Chat"])

    # --- TAB 1: OVERVIEW (NEW) ---
    with tab1:
        st.header("🏢 Business Overview")
        
        # 1. METRICS ROW
        rev, orders, avg_day, avg_week, aov = get_overview_metrics(df)
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Total Revenue", f"₹{rev:,.0f}")
        c2.metric("Total Orders", orders)
        c3.metric("Avg Rev/Day", f"₹{avg_day:,.0f}")
        c4.metric("Avg Rev/Week", f"₹{avg_week:,.0f}")
        c5.metric("Avg Order Value", f"₹{aov:.0f}")
        st.divider()

        # 2. GRAPHS ROW (Peak Hours & Peak Days)
        g1, g2 = st.columns(2)
        with g1:
            st.subheader("⌚ Peak Hours Graph")
            if 'Hour' in df.columns:
                hourly = df.groupby('Hour')['TotalAmount'].sum().reset_index()
                st.bar_chart(hourly.set_index('Hour'))
            else:
                st.warning("No Time data found.")
        
        with g2:
            st.subheader("📅 Peak Days Graph")
            daily_peak = analyze_peak_days(df)
            st.bar_chart(daily_peak.set_index('DayOfWeek'))

        st.divider()

        # 3. LISTS ROW (Peak Items & High Revenue Days)
        l1, l2 = st.columns(2)
        with l1:
            peak_items, top_hours = analyze_peak_hour_items(df)
            st.subheader(f"🔥 Items sold in Peak Hours ({top_hours})")
            st.dataframe(peak_items, hide_index=True)
            
        with l2:
            st.subheader("💰 High Revenue Days")
            top_days = df.groupby('Date')['TotalAmount'].sum().sort_values(ascending=False).head(5).reset_index()
            top_days['Date'] = top_days['Date'].dt.date
            st.dataframe(top_days, hide_index=True)

        st.divider()

        # 4. CONTRIBUTIONS ROW
        cat_cont, item_cont, star_items = get_contribution_lists(df)
        
        c_col1, c_col2 = st.columns(2)
        with c_col1:
            st.subheader("📂 Category Contribution %")
            st.dataframe(cat_cont[['Category', 'TotalAmount', 'Contribution %']], hide_index=True)
            
            st.subheader("⭐ Top 20 Star Items")
            st.dataframe(star_items[['ItemName', 'TotalAmount', 'Contribution %']], hide_index=True)

        with c_col2:
            st.subheader("🍽️ Item Contribution (by Category)")
            st.dataframe(item_cont[['Category', 'ItemName', 'TotalAmount', 'Contribution %']], height=500, hide_index=True)

    # --- TAB 2: PARETO ANALYSIS (MOVED HERE) ---
    with tab2:
        st.header("🏆 Pareto Analysis (80/20 Rule)")
        pareto_df = analyze_pareto(df)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("##### Contribution of Categories to Top 80%")
            cat_pie = px.pie(pareto_df, values='TotalAmount', names='Category', hole=0.4)
            st.plotly_chart(cat_pie, use_container_width=True)
        
        with col2:
            st.markdown("##### Items driving 80% of Business")
            st.dataframe(pareto_df[['ItemName', 'Category', 'TotalAmount', 'CumPerc']], height=500)

    # --- TAB 3: TIME SERIES ---
    with tab3:
        st.header("📅 Daily Trends")
        plot_time_series_top5(df)

    # --- TAB 4: BASKET & CORR ---
    with tab4:
        st.header("🛒 Basket & Correlations")
        b1, b2 = st.columns(2)
        with b1:
            st.subheader("Item Combos (>60%)")
            st.dataframe(get_basket_rules(df, 'ItemName', 0.6))
        with b2:
            st.subheader("Category Combos (>60%)")
            st.dataframe(get_basket_rules(df, 'Category', 0.6))
        
        st.divider()
        if st.checkbox("Show Correlation Heatmap"):
            st.pyplot(correlation_heatmap(df, 'Category'))

    # --- TAB 5: FORECAST ---
    with tab5:
        st.header("🔮 Demand Forecast (30 Days)")
        forecast_df = forecast_demand(df)
        sel_cat = st.selectbox("Select Category", df['Category'].unique())
        cat_items = df[df['Category'] == sel_cat]['ItemName'].unique()
        subset = forecast_df[forecast_df['ItemName'].isin(cat_items)]
        if not subset.empty:
            st.dataframe(subset.pivot(index='ItemName', columns='Date', values='Predicted_Qty'))

    # --- TAB 6: AI CHAT ---
    with tab6:
        st.subheader("🤖 Chat with Manager")
        if "messages" not in st.session_state: st.session_state.messages = []
        for msg in st.session_state.messages: st.chat_message(msg["role"]).write(msg["content"])
        if prompt := st.chat_input("Ask about data..."):
            st.chat_message("user").write(prompt)
            st.session_state.messages.append({"role": "user", "content": prompt})
            try:
                llm = ChatOpenAI(model="gpt-4o-mini", api_key=st.secrets["OPENAI_API_KEY"])
                response = llm.invoke([SystemMessage(content="Restaurant Analyst"), HumanMessage(content=f"Data: Rev {rev}. Q: {prompt}")])
                st.chat_message("assistant").write(response.content)
                st.session_state.messages.append({"role": "assistant", "content": response.content})
            except: st.error("Check API Key")

else:
    st.info("👋 Upload data to begin.")
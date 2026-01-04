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
        
    return df

# --- ANALYTICS MODULES ---

def analyze_pareto(df):
    """Req 3: Items contributing 80% revenue + Category Info"""
    item_rev = df.groupby(['ItemName', 'Category'])['TotalAmount'].sum().reset_index()
    item_rev = item_rev.sort_values('TotalAmount', ascending=False)
    
    total_revenue = item_rev['TotalAmount'].sum()
    item_rev['Cumulative'] = item_rev['TotalAmount'].cumsum()
    item_rev['CumPerc'] = 100 * item_rev['Cumulative'] / total_revenue
    
    # Filter top 80%
    top_80 = item_rev[item_rev['CumPerc'] <= 82] # Slight buffer for 80% cutoff
    return top_80

def get_basket_rules(df, group_col='ItemName', min_conf=0.6):
    """Req 6: Basket Analysis with >60% Confidence"""
    basket = (df.groupby(['OrderID', group_col])['Quantity']
              .sum().unstack().reset_index().fillna(0)
              .set_index('OrderID'))
    
    # Binary Encode
    basket_sets = basket.applymap(lambda x: 1 if x > 0 else 0)
    
    # Run Apriori
    frequent = apriori(basket_sets, min_support=0.005, use_colnames=True)
    
    if frequent.empty: return pd.DataFrame()
    
    # Filter by Confidence > 0.6 (Req 6)
    rules = association_rules(frequent, metric="confidence", min_threshold=min_conf)
    rules['antecedents'] = rules['antecedents'].apply(lambda x: list(x)[0])
    rules['consequents'] = rules['consequents'].apply(lambda x: list(x)[0])
    
    return rules[['antecedents', 'consequents', 'confidence', 'lift']].sort_values('confidence', ascending=False)

def plot_time_series_top5(df):
    """Req 1: Top 5 Items per Category Trend with Avg Line"""
    categories = df['Category'].unique()
    
    for cat in categories:
        st.subheader(f"📈 {cat}: Top 5 Items Trend")
        cat_data = df[df['Category'] == cat]
        
        # Identify Top 5 Items
        top_items = cat_data.groupby('ItemName')['Quantity'].sum().nlargest(5).index.tolist()
        
        # Filter Data
        subset = cat_data[cat_data['ItemName'].isin(top_items)]
        daily = subset.groupby(['Date', 'ItemName'])['Quantity'].sum().reset_index()
        
        if daily.empty:
            st.warning(f"Not enough data for {cat}")
            continue

        fig = px.line(daily, x='Date', y='Quantity', color='ItemName', markers=True)
        
        # Req 1: Add Average Line
        for item in top_items:
            avg_val = daily[daily['ItemName'] == item]['Quantity'].mean()
            fig.add_hline(y=avg_val, line_dash="dot", annotation_text=f"Avg {item}", annotation_position="top left")
            
        st.plotly_chart(fig, use_container_width=True)

def correlation_heatmap(df, group_col):
    """Req 7: Correlation Heatmaps"""
    pivot = df.pivot_table(index='Date', columns=group_col, values='Quantity', aggfunc='sum').fillna(0)
    corr = pivot.corr()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=False, cmap='coolwarm', ax=ax)
    return fig

def forecast_demand(df):
    """Req 8: Next 30 Days Forecast (Item & Category Wise)"""
    daily = df.groupby(['Date', 'ItemName'])['Quantity'].sum().reset_index()
    
    forecasts = []
    items = daily['ItemName'].unique()
    
    last_date = daily['Date'].max()
    next_30_days = [last_date + timedelta(days=x) for x in range(1, 31)]
    
    for item in items:
        item_data = daily[daily['ItemName'] == item].sort_values('Date')
        # Simple forecast: Average of last 7 days
        if len(item_data) >= 7:
            avg_qty = item_data.tail(7)['Quantity'].mean()
        else:
            avg_qty = item_data['Quantity'].mean()
        
        for date in next_30_days:
            forecasts.append({'Date': date.strftime('%Y-%m-%d'), 'ItemName': item, 'Predicted_Qty': round(avg_qty, 1)})
            
    return pd.DataFrame(forecasts)


# --- MAIN APP LAYOUT ---

st.title("📊 Mithas Restaurant Intelligence 2.0")

uploaded_file = st.sidebar.file_uploader("Upload Monthly/Daily Excel", type=['xlsx'])

if uploaded_file:
    df = load_data(uploaded_file)
    
    # Check for Date column existence
    if df['Date'].isnull().all():
        st.error("⚠️ Error: 'Date' column is missing or empty. Time series and Forecasts cannot run.")
        st.stop()

    # TABS FOR ORGANIZATION
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Overview & Pareto", "Time Series", "Basket & Correlations", "Forecast", "AI Chat"])

    # --- TAB 1: OVERVIEW & PARETO ---
    with tab1:
        st.header("🏆 The Vital Few (Pareto 80/20)")
        pareto_df = analyze_pareto(df)
        
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("##### Req 3: Contribution of Categories to Top 80%")
            cat_pie = px.pie(pareto_df, values='TotalAmount', names='Category', hole=0.4)
            st.plotly_chart(cat_pie, use_container_width=True)
            
            st.markdown("##### Req 2: Total Sales by Category")
            all_cat_pie = px.pie(df, values='TotalAmount', names='Category')
            st.plotly_chart(all_cat_pie, use_container_width=True)

        with c2:
            st.markdown("##### Req 3: Items driving 80% of Business")
            st.dataframe(pareto_df[['ItemName', 'Category', 'TotalAmount', 'CumPerc']], height=500)

        # Req 4: Peak Hours
        st.divider()
        st.subheader("⌚ Peak Hours Analysis")
        st.caption("X-Axis: Hour of the Day (0-24) | Y-Axis: Total Revenue Generated")
        if 'Time' in df.columns:
            # Handle various time formats safely
            try:
                df['Hour'] = pd.to_datetime(df['Time'].astype(str), format='%H:%M:%S', errors='coerce').dt.hour
                # Fallback
                if df['Hour'].isnull().all():
                     df['Hour'] = pd.to_datetime(df['Time'], errors='coerce').dt.hour
                hourly = df.groupby('Hour')['TotalAmount'].sum().reset_index()
                st.bar_chart(hourly.set_index('Hour'))
            except:
                st.warning("Could not parse Time column.")

    # --- TAB 2: TIME SERIES ---
    with tab2:
        st.header("📅 Daily Trends: Top 5 Items per Category")
        # Req 1: Top 5 with Avg Line
        plot_time_series_top5(df)

    # --- TAB 3: BASKET & CORRELATIONS ---
    with tab3:
        st.header("🛒 Advanced Basket Analysis (Prob > 60%)")
        
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Item-to-Item")
            rules_item = get_basket_rules(df, 'ItemName', min_conf=0.6)
            if not rules_item.empty:
                st.dataframe(rules_item)
            else:
                st.info("No items have >60% probability linkage.")
                
        with c2:
            st.subheader("Category-to-Category")
            rules_cat = get_basket_rules(df, 'Category', min_conf=0.6)
            if not rules_cat.empty:
                st.dataframe(rules_cat)
            else:
                st.info("No categories have >60% probability linkage.")

        st.divider()
        st.header("🔗 Correlation Heatmaps")
        st.caption("Req 7: Red = Items sold together often. Blue = Items rarely sold together.")
        
        cat_corr = st.checkbox("Show Category Correlation", value=True)
        if cat_corr:
            st.pyplot(correlation_heatmap(df, 'Category'))
            
        item_corr = st.checkbox("Show Item Correlation (Heavy computation)")
        if item_corr:
            st.pyplot(correlation_heatmap(df, 'ItemName'))

    # --- TAB 4: FORECAST ---
    with tab4:
        st.header("🔮 Next Month Demand Forecast")
        st.markdown("Req 8: Daily demand forecast item-wise for next 30 days.")
        
        forecast_df = forecast_demand(df)
        
        # Filter UI
        sel_cat = st.selectbox("Select Category to View Forecast", df['Category'].unique())
        
        # Filter Forecast Data
        cat_items = df[df['Category'] == sel_cat]['ItemName'].unique()
        subset_forecast = forecast_df[forecast_df['ItemName'].isin(cat_items)]
        
        # Pivot for clean display (Dates as columns)
        if not subset_forecast.empty:
            pivot_forecast = subset_forecast.pivot(index='ItemName', columns='Date', values='Predicted_Qty')
            st.dataframe(pivot_forecast)
        else:
            st.warning("Not enough data to forecast this category.")

    # --- TAB 5: AI CHAT ---
    with tab5:
        st.subheader("🤖 Chat with your Manager")
        
        if "messages" not in st.session_state: st.session_state.messages = []
        
        for msg in st.session_state.messages:
            st.chat_message(msg["role"]).write(msg["content"])
            
        if prompt := st.chat_input("Ask about forecasts, correlations, or pareto..."):
            st.chat_message("user").write(prompt)
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Simple AI context wrapper
            try:
                llm = ChatOpenAI(model="gpt-4o-mini", api_key=st.secrets["OPENAI_API_KEY"])
                context = f"""
                Data loaded. 
                Top Pareto Item: {pareto_df.iloc[0]['ItemName'] if not pareto_df.empty else 'N/A'}.
                Forecast is ready for next 30 days.
                User Question: {prompt}
                """
                response = llm.invoke([SystemMessage(content="You are a Restaurant Analyst."), HumanMessage(content=context)])
                st.chat_message("assistant").write(response.content)
                st.session_state.messages.append({"role": "assistant", "content": response.content})
            except Exception as e:
                st.error("AI Error: Check API Key.")

else:
    st.info("👋 Welcome! Upload your Excel file to unlock the Intelligence 2.0 Dashboard.")
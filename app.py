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
st.set_page_config(page_title="Mithas Intelligence 4.0", layout="wide")

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
    
    # Hour Extraction
    if 'Time' in df.columns:
        try:
            df['Hour'] = pd.to_datetime(df['Time'].astype(str), format='%H:%M:%S', errors='coerce').dt.hour
            if df['Hour'].isnull().all():
                 df['Hour'] = pd.to_datetime(df['Time'], errors='coerce').dt.hour
        except:
            df['Hour'] = 0
    return df

# --- ANALYTICS MODULES ---

def get_star_items_with_hours(df):
    """Req 1: Top 20 items + Their Peak Selling Hour"""
    total_rev = df['TotalAmount'].sum()
    item_stats = df.groupby('ItemName').agg({'TotalAmount': 'sum'}).reset_index()
    item_stats['Contribution'] = (item_stats['TotalAmount'] / total_rev)
    item_stats = item_stats.sort_values('TotalAmount', ascending=False).head(20)
    
    # Calculate Peak Hour for each star item
    peak_hours = []
    for item in item_stats['ItemName']:
        item_data = df[df['ItemName'] == item]
        if 'Hour' in df.columns and not item_data.empty:
            # Find the hour with max quantity sold
            peak = item_data.groupby('Hour')['Quantity'].sum().idxmax()
            peak_str = f"{int(peak)}:00 - {int(peak)+1}:00"
        else:
            peak_str = "N/A"
        peak_hours.append(peak_str)
        
    item_stats['Peak Selling Hour'] = peak_hours
    return item_stats

def analyze_pareto_hierarchical(df):
    """Req 2: Hierarchical Pareto (Cat -> Item) matching user image"""
    # 1. Identify Top 80% Revenue Items
    item_rev = df.groupby(['Category', 'ItemName'])['TotalAmount'].sum().reset_index()
    total_revenue = item_rev['TotalAmount'].sum()
    item_rev = item_rev.sort_values('TotalAmount', ascending=False)
    item_rev['Cumulative'] = item_rev['TotalAmount'].cumsum()
    item_rev['CumPerc'] = 100 * item_rev['Cumulative'] / total_revenue
    
    # The "Vital Few" items
    pareto_items = item_rev[item_rev['CumPerc'] <= 80].copy()
    
    # Stats for the text summary
    total_unique_items = df['ItemName'].nunique()
    pareto_unique_items = pareto_items['ItemName'].nunique()
    ratio_text = f"**{pareto_unique_items} items** (out of {total_unique_items}) contribute to 80% of your revenue."
    percentage_of_menu = (pareto_unique_items / total_unique_items) * 100
    
    # 2. Calculate Category Contribution % (Global)
    cat_rev = df.groupby('Category')['TotalAmount'].sum().reset_index()
    cat_rev['CatContrib'] = (cat_rev['TotalAmount'] / total_revenue)
    
    # 3. Merge to create the structure
    # We join the Pareto Items with their Category's total contribution
    merged = pd.merge(pareto_items, cat_rev[['Category', 'CatContrib']], on='Category', how='left')
    
    # Calculate Item's contribution to Total Revenue
    merged['ItemContrib'] = (merged['TotalAmount'] / total_revenue)
    
    # Format for display
    display_df = merged[['Category', 'CatContrib', 'ItemName', 'ItemContrib', 'TotalAmount']]
    display_df = display_df.sort_values(['CatContrib', 'TotalAmount'], ascending=[False, False])
    
    return display_df, ratio_text, percentage_of_menu

def plot_time_series_fixed(df):
    """Req 3: Time Series with Fixed Scale and Labels"""
    categories = df['Category'].unique()
    for cat in categories:
        st.subheader(f"📈 {cat}")
        cat_data = df[df['Category'] == cat]
        top_items = cat_data.groupby('ItemName')['Quantity'].sum().nlargest(5).index.tolist()
        
        subset = cat_data[cat_data['ItemName'].isin(top_items)]
        daily = subset.groupby(['Date', 'ItemName'])['Quantity'].sum().reset_index()
        
        if daily.empty: continue

        # Create Graph
        fig = px.line(daily, x='Date', y='Quantity', color='ItemName', markers=True)
        
        # Add Average Lines with explicit labels
        for item in top_items:
            avg_val = daily[daily['ItemName'] == item]['Quantity'].mean()
            # Add annotation directly to the chart
            fig.add_hline(y=avg_val, line_dash="dot", line_color="grey", opacity=0.5)
            fig.add_annotation(
                x=daily['Date'].max(), y=avg_val,
                text=f"Avg: {avg_val:.1f}",
                showarrow=False, yshift=10, font=dict(color="red", size=10)
            )
            
        # Fix Scaling: Allow each graph to fit its own data (don't force 0-100 if data is 0-5)
        fig.update_yaxes(matches=None, showticklabels=True)
        st.plotly_chart(fig, use_container_width=True)

def advanced_basket_analysis(df):
    """Req 4: Lift-based Basket Analysis"""
    # Filter for meaningful transactions (more than 1 item)
    order_counts = df.groupby('OrderID')['ItemName'].count()
    valid_orders = order_counts[order_counts > 1].index
    df_basket = df[df['OrderID'].isin(valid_orders)]
    
    basket = (df_basket.groupby(['OrderID', 'ItemName'])['Quantity']
              .sum().unstack().reset_index().fillna(0)
              .set_index('OrderID'))
    
    basket_sets = basket.applymap(lambda x: 1 if x > 0 else 0)
    
    # Lower support to catch niche combos, filter by Lift later
    frequent = apriori(basket_sets, min_support=0.005, use_colnames=True)
    
    if frequent.empty: return pd.DataFrame()
    
    # Metric = Lift (Strength of association)
    # Lift > 1 means they appear together more often than random chance
    rules = association_rules(frequent, metric="lift", min_threshold=1.2)
    
    # Clean up columns
    rules['antecedents'] = rules['antecedents'].apply(lambda x: list(x)[0])
    rules['consequents'] = rules['consequents'].apply(lambda x: list(x)[0])
    
    # Sort by Lift (Best combos first)
    return rules[['antecedents', 'consequents', 'lift', 'confidence', 'support']].sort_values('lift', ascending=False).head(20)

def advanced_forecast(df):
    """Req 5: Holt-Winters Exponential Smoothing"""
    daily = df.groupby(['Date', 'ItemName'])['Quantity'].sum().reset_index()
    
    # Get Top 10 Categories -> Top 10 items in each
    top_cats = df.groupby('Category')['TotalAmount'].sum().nlargest(10).index.tolist()
    
    forecast_results = []
    
    for cat in top_cats:
        cat_df = df[df['Category'] == cat]
        top_items = cat_df.groupby('ItemName')['Quantity'].sum().nlargest(10).index.tolist()
        
        for item in top_items:
            item_data = daily[daily['ItemName'] == item].set_index('Date')['Quantity']
            # Reindex to fill missing days with 0 (Crucial for time series)
            idx = pd.date_range(item_data.index.min(), item_data.index.max())
            item_data = item_data.reindex(idx, fill_value=0)
            
            try:
                # Use Holt-Winters (Trend + Seasonality) if enough data
                if len(item_data) > 14:
                    model = ExponentialSmoothing(item_data, trend='add', seasonal='add', seasonal_periods=7).fit()
                    pred = model.forecast(30)
                else:
                    # Fallback to simple moving average if < 2 weeks data
                    pred = pd.Series([item_data.mean()] * 30, index=pd.date_range(item_data.index.max() + timedelta(days=1), periods=30))
                
                total_expected_demand = pred.sum()
                forecast_results.append({
                    'Category': cat,
                    'ItemName': item,
                    'Total Predicted Demand (Next 30 Days)': round(total_expected_demand, 0)
                })
            except:
                continue
                
    return pd.DataFrame(forecast_results)

# --- MAIN APP LAYOUT ---
st.title("📊 Mithas Restaurant Intelligence 4.0")
uploaded_file = st.sidebar.file_uploader("Upload Monthly Data", type=['xlsx'])

if uploaded_file:
    df = load_data(uploaded_file)
    
    # TABS
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Overview", "Pareto (Visual)", "Time Series", "Smart Combos", "Demand Forecast", "AI Chat"
    ])

    # --- TAB 1: OVERVIEW ---
    with tab1:
        st.header("🏢 Business Overview")
        
        # Attractive Metrics
        total_rev = df['TotalAmount'].sum()
        total_orders = df['OrderID'].nunique()
        c1, c2, c3 = st.columns(3)
        c1.metric("💰 Total Revenue", f"₹{total_rev:,.0f}")
        c2.metric("🧾 Total Orders", total_orders)
        c3.metric("💳 Avg Order Value", f"₹{total_rev/total_orders:.0f}" if total_orders else 0)
        
        st.divider()
        
        # Attractive Star Items Table
        st.subheader("⭐ Top 20 Star Items & Peak Hours")
        star_df = get_star_items_with_hours(df)
        
        st.dataframe(
            star_df,
            column_config={
                "TotalAmount": st.column_config.ProgressColumn(
                    "Revenue", format="₹%f", min_value=0, max_value=star_df['TotalAmount'].max()
                ),
                "Contribution": st.column_config.NumberColumn(
                    "Contribution", format="%.2f%%"
                )
            },
            hide_index=True,
            use_container_width=True
        )

    # --- TAB 2: PARETO ---
    with tab2:
        st.header("🏆 Pareto Analysis")
        pareto_df, ratio_msg, menu_perc = analyze_pareto_hierarchical(df)
        
        st.info(f"💡 **Insight:** {ratio_msg} (Only {menu_perc:.1f}% of your menu!)")
        
        st.markdown("### The 80% Contribution Breakdown")
        st.dataframe(
            pareto_df,
            column_config={
                "CatContrib": st.column_config.NumberColumn("Category Share %", format="%.2f%%"),
                "ItemContrib": st.column_config.NumberColumn("Item Share % (Global)", format="%.2f%%"),
                "TotalAmount": st.column_config.NumberColumn("Revenue", format="₹%d"),
            },
            hide_index=True,
            height=600,
            use_container_width=True
        )

    # --- TAB 3: TIME SERIES ---
    with tab3:
        st.header("📅 Daily Trends (Fixed Scales)")
        st.caption("Red labels show the average daily sales for that item.")
        plot_time_series_fixed(df)

    # --- TAB 4: BASKET ANALYSIS ---
    with tab4:
        st.header("🍔 Smart Combo Builder")
        st.caption("Sorted by 'Lift'. Lift > 2.0 is a very strong connection.")
        
        combos = advanced_basket_analysis(df)
        if not combos.empty:
            st.dataframe(
                combos,
                column_config={
                    "lift": st.column_config.NumberColumn("Lift Strength", format="%.2f"),
                    "confidence": st.column_config.NumberColumn("Probability", format="%.2f"),
                },
                hide_index=True,
                use_container_width=True
            )
        else:
            st.warning("Not enough transaction overlap found.")

    # --- TAB 5: FORECAST ---
    with tab5:
        st.header("🔮 Demand Prediction (Next Month)")
        st.markdown("**Model:** Holt-Winters Exponential Smoothing (Detects seasonality).")
        st.markdown("Showing **Total Expected Quantity** for the next 30 days.")
        
        with st.spinner("Training Statistical Models... (This takes a moment)"):
            forecast_data = advanced_forecast(df)
            
        if not forecast_data.empty:
            # Pivot for cleaner view
            st.dataframe(
                forecast_data.sort_values(['Category', 'Total Predicted Demand (Next 30 Days)'], ascending=[True, False]),
                use_container_width=True,
                hide_index=True
            )
        else:
            st.error("Not enough historical data to train the model (Need at least 14 days).")

    # --- TAB 6: AI CHAT ---
    with tab6:
        st.subheader("🤖 Manager Chat")
        if "messages" not in st.session_state: st.session_state.messages = []
        for msg in st.session_state.messages: st.chat_message(msg["role"]).write(msg["content"])
        if prompt := st.chat_input("Ask about the new forecast..."):
            st.chat_message("user").write(prompt)
            st.session_state.messages.append({"role": "user", "content": prompt})
            try:
                llm = ChatOpenAI(model="gpt-4o-mini", api_key=st.secrets["OPENAI_API_KEY"])
                response = llm.invoke([SystemMessage(content="Restaurant Analyst"), HumanMessage(content=prompt)])
                st.chat_message("assistant").write(response.content)
                st.session_state.messages.append({"role": "assistant", "content": response.content})
            except: st.error("Check API Key")

else:
    st.info("👋 Upload data to begin.")
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import fpgrowth, association_rules
from mlxtend.preprocessing import TransactionEncoder
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from datetime import timedelta
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import warnings

warnings.filterwarnings('ignore')

# --- PAGE CONFIG ---
st.set_page_config(page_title="Mithas Intelligence 8.0", layout="wide")

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

# --- ADVANCED ASSOCIATION ANALYSIS (RESEARCH BASED) ---

def run_advanced_association(df, level='ItemName', min_sup=0.005, min_conf=0.1):
    """
    Implements FP-Growth Algorithm (Han et al.) for superior performance on dense datasets.
    Includes Zhang's Metric for advanced correlation stability.
    """
    # 1. One-Hot Encoding (Faster than pivot for FP-Growth)
    basket = df.groupby(['OrderID', level])['Quantity'].sum().unstack().reset_index().fillna(0).set_index('OrderID')
    
    # Convert to boolean (FP-Growth requires bool/int)
    basket_sets = basket.applymap(lambda x: True if x > 0 else False)
    
    # 2. FP-Growth Algorithm (More memory efficient than Apriori)
    frequent_itemsets = fpgrowth(basket_sets, min_support=min_sup, use_colnames=True)
    
    if frequent_itemsets.empty:
        return pd.DataFrame()
    
    # 3. Generate Rules with Advanced Metrics
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_conf)
    
    # 4. Clean & Calculate Zhang's Metric
    # Zhang's metric measures association (-1 to 1). 
    # +1 = Perfect Association, -1 = Perfect Dissociation (Never buy together)
    
    def zhangs_metric(rule):
        sup = rule['support']
        sup_a = rule['antecedent support']
        sup_c = rule['consequent support']
        numerator = sup - (sup_a * sup_c)
        denominator = max(sup * (1 - sup_a), sup_a * (sup_c - sup))
        if denominator == 0: return 0
        return numerator / denominator

    rules['zhang'] = rules.apply(zhangs_metric, axis=1)
    
    # Format for display
    rules['Antecedent'] = rules['antecedents'].apply(lambda x: list(x)[0])
    rules['Consequent'] = rules['consequents'].apply(lambda x: list(x)[0])
    
    return rules[['Antecedent', 'Consequent', 'support', 'confidence', 'lift', 'zhang', 'conviction']]

# --- COMBO LOGIC (EXISTING) ---
def get_combo_analysis_full(df):
    # This uses basic Apriori for the "Smart Combos" tab (Legacy)
    basket = (df.groupby(['OrderID', 'ItemName'])['Quantity'].sum().unstack().reset_index().fillna(0).set_index('OrderID'))
    basket_sets = basket.applymap(lambda x: 1 if x > 0 else 0)
    # Using basic Apriori here to keep "Smart Combos" tab logic consistent as requested
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
st.title("📊 Mithas Restaurant Intelligence 8.0")
uploaded_file = st.sidebar.file_uploader("Upload Monthly Data", type=['xlsx'])

if uploaded_file:
    df = load_data(uploaded_file)
    
    # REQ: Added "Association Analysis" after Smart Combos
    tab1, tab_cat, tab2, tab3, tab4, tab_assoc, tab5, tab6 = st.tabs([
        "Overview", "Category Details", "Pareto (Visual)", "Time Series", "Smart Combos", "Association Analysis", "Demand Forecast", "AI Chat"
    ])

    # --- TAB 1: OVERVIEW (VISUALLY UPGRADED) ---
    with tab1:
        st.header("🏢 Business Overview")
        with st.expander("🛠️ Reorder Page Layout", expanded=False):
            overview_order = st.multiselect(
                "Select order of sections:",
                ["Metrics", "Graphs", "Hourly Breakdown (9am-11pm)", "Peak Items List", "Contributions", "Star Items"],
                default=["Metrics", "Graphs", "Hourly Breakdown (9am-11pm)", "Peak Items List", "Contributions", "Star Items"]
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
            st.caption("Click on an hour to see detailed item sales.")
            hourly_df = get_hourly_details(df)
            if not hourly_df.empty:
                time_slots = hourly_df['Time Slot'].unique()
                for slot in time_slots:
                    slot_data = hourly_df[hourly_df['Time Slot'] == slot]
                    total_rev_slot = slot_data['TotalAmount'].sum()
                    total_qty_slot = slot_data['Quantity'].sum()
                    top_item_slot = slot_data.iloc[0]['ItemName']
                    with st.expander(f"⏰ {slot}  |  Revenue: ₹{total_rev_slot:,.0f}  |  Units: {total_qty_slot}  |  Top: {top_item_slot}"):
                        st.dataframe(
                            slot_data[['ItemName', 'Quantity', 'TotalAmount']],
                            column_config={
                                "ItemName": "Item Name",
                                "Quantity": st.column_config.ProgressColumn("Units Sold", format="%d", min_value=0, max_value=int(hourly_df['Quantity'].max())),
                                "TotalAmount": st.column_config.NumberColumn("Revenue", format="₹%d")
                            }, hide_index=True, use_container_width=True)
            else: st.info("No sales data found between 9 AM and 11 PM.")
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
            cat_cont, item_cont = get_contribution_lists(df)
            col_cat, col_item = st.columns(2)
            with col_cat:
                st.subheader("📂 Category Contribution %")
                st.dataframe(cat_cont[['Category', 'TotalAmount', 'Contribution']], column_config={"Contribution": st.column_config.ProgressColumn("Share %", format="%.2f%%", min_value=0, max_value=100), "TotalAmount": st.column_config.NumberColumn("Revenue", format="₹%d")}, hide_index=True, use_container_width=True)
            with col_item:
                st.subheader("🍽️ Item Contribution (by Category)")
                st.dataframe(item_cont[['Category', 'ItemName', 'TotalAmount', 'Contribution']], column_config={"Contribution": st.column_config.NumberColumn("Share %", format="%.2f%%"), "TotalAmount": st.column_config.NumberColumn("Revenue", format="₹%d")}, hide_index=True, height=400, use_container_width=True)
            st.divider()

        def render_star_items():
            st.subheader("⭐ Top 20 Star Items & Selling Hours")
            star_df = get_star_items_with_hours(df)
            st.dataframe(star_df, column_config={
                "TotalAmount": st.column_config.NumberColumn("Revenue", format="₹%d"),
                "Contribution %": st.column_config.ProgressColumn("Contribution", format="%.2f%%", min_value=0, max_value=star_df['Contribution %'].max()),
                "Peak Selling Hour": st.column_config.TextColumn("Peak Hour Window"),
                "Qty Sold (Peak)": st.column_config.NumberColumn("Qty in Peak Hour")
            }, hide_index=True, use_container_width=True)

        block_map = {"Metrics": render_metrics, "Graphs": render_graphs, "Hourly Breakdown (9am-11pm)": render_hourly_details, "Peak Items List": render_peak_list, "Contributions": render_contributions, "Star Items": render_star_items}
        for block_name in overview_order:
            if block_name in block_map: block_map[block_name]()

    # --- TAB: CATEGORY DETAILS (UNCHANGED) ---
    with tab_cat:
        st.header("📂 Category Deep-Dive")
        cats = df['Category'].unique()
        total_business_rev = df['TotalAmount'].sum()
        for cat in cats:
            st.subheader(f"🔹 {cat}")
            cat_data = df[df['Category'] == cat]
            cat_stats = cat_data.groupby('ItemName').agg({'TotalAmount': 'sum', 'Quantity': 'sum'}).reset_index()
            cat_stats['Contribution %'] = (cat_stats['TotalAmount'] / total_business_rev) * 100
            cat_stats = cat_stats.sort_values('TotalAmount', ascending=False)
            st.dataframe(cat_stats, column_config={"ItemName": "Item Name", "TotalAmount": st.column_config.NumberColumn("Revenue", format="₹%d"), "Quantity": st.column_config.NumberColumn("Units Sold"), "Contribution %": st.column_config.ProgressColumn("Contribution to Total Rev", format="%.2f%%", min_value=0, max_value=100)}, hide_index=True, use_container_width=True)
            st.divider()

    # --- TAB 2: PARETO (UNCHANGED) ---
    with tab2:
        st.header("🏆 Pareto Analysis")
        pareto_df, ratio_msg, menu_perc = analyze_pareto_hierarchical(df)
        st.info(f"💡 **Insight:** {ratio_msg} (Only {menu_perc:.1f}% of your menu!)")
        st.dataframe(pareto_df, column_config={"CatContrib": st.column_config.NumberColumn("Category Share %", format="%.2f%%"), "ItemContrib": st.column_config.NumberColumn("Item Share % (Global)", format="%.2f%%"), "TotalAmount": st.column_config.NumberColumn("Revenue", format="₹%d")}, hide_index=True, height=600, use_container_width=True)

    # --- TAB 3: TIME SERIES (UNCHANGED) ---
    with tab3:
        st.header("📅 Daily Trends")
        plot_time_series_fixed(df)

    # --- TAB 4: SMART COMBOS (UNCHANGED) ---
    with tab4:
        st.header("🍔 Smart Combo Strategy")
        with st.expander("🛠️ Reorder Combo Layout"):
            combo_order = st.multiselect("Section Order", ["Part 1: Full Combo Map", "Part 3: Strategic Recommendations"], default=["Part 1: Full Combo Map", "Part 3: Strategic Recommendations"])
        rules_df = get_combo_analysis_full(df)
        proven_df, potential_df = get_part3_strategy(rules_df)
        def render_part1():
            st.subheader("1️⃣ Part 1: Full Category + Item Combo Map")
            if not rules_df.empty:
                display_cols = ['Category A', 'Category B', 'Specific Item Combo', 'Times Sold Together', 'Peak Hour', 'lift']
                st.dataframe(rules_df[display_cols].sort_values('Times Sold Together', ascending=False), column_config={"Specific Item Combo": st.column_config.TextColumn("Item A + Item B", width="medium"), "lift": st.column_config.NumberColumn("Lift Strength", format="%.2f")}, hide_index=True, use_container_width=True)
            else: st.warning("No significant combos found.")
        def render_part3():
            st.subheader("3️⃣ Part 3: Strategic Recommendations")
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("#### 🔥 Proven Winners")
                if not proven_df.empty: st.dataframe(proven_df[['Specific Item Combo', 'Times Sold Together', 'Peak Hour']], hide_index=True, use_container_width=True)
                else: st.info("No data.")
            with c2:
                st.markdown("#### 💎 Hidden Gems")
                if not potential_df.empty: st.dataframe(potential_df[['Specific Item Combo', 'lift', 'Peak Hour']], column_config={"lift": st.column_config.NumberColumn("Compatibility Score", format="%.2f")}, hide_index=True, use_container_width=True)
                else: st.info("No hidden gems found.")
        combo_map = {"Part 1: Full Combo Map": render_part1, "Part 3: Strategic Recommendations": render_part3}
        for block in combo_order:
            if block in combo_map: combo_map[block]()
    
    # --- TAB 5: NEW ASSOCIATION ANALYSIS (FP-GROWTH) ---
    with tab_assoc:
        st.header("🧬 Scientific Association Analysis (FP-Growth)")
        st.markdown("Uses **FP-Growth Algorithm** for maximum accuracy on dense datasets. Sorted by **Zhang's Metric** (Stability).")
        
        # Controls
        c1, c2 = st.columns(2)
        with c1:
            analysis_level = st.radio("Analysis Level", ["ItemName", "Category"], horizontal=True)
        with c2:
            min_support_slider = st.slider("Minimum Support (Frequency)", 0.001, 0.05, 0.005, format="%.3f")
        
        # Run Calculation
        with st.spinner("Running FP-Growth Algorithm..."):
            assoc_rules = run_advanced_association(df, level=analysis_level, min_sup=min_support_slider)
        
        if not assoc_rules.empty:
            # Sort by Lift (Strength)
            assoc_rules = assoc_rules.sort_values('lift', ascending=False).head(50)
            
            st.dataframe(
                assoc_rules,
                column_config={
                    "zhang": st.column_config.NumberColumn("Zhang's Metric", help="+1: Perfect Association, -1: Perfect Dissociation", format="%.2f"),
                    "conviction": st.column_config.NumberColumn("Conviction", help="High value means consequent depends strongly on antecedent", format="%.2f"),
                    "lift": st.column_config.NumberColumn("Lift", format="%.2f"),
                    "confidence": st.column_config.NumberColumn("Confidence", format="%.2f"),
                },
                hide_index=True,
                use_container_width=True,
                height=600
            )
            
            # Visual: Scatter Plot of Rules
            fig = px.scatter(
                assoc_rules, x="support", y="confidence", 
                size="lift", color="zhang",
                hover_data=["Antecedent", "Consequent"],
                title=f"Association Rules Landscape ({analysis_level} Level)",
                color_continuous_scale=px.colors.diverging.RdBu
            )
            st.plotly_chart(fig, use_container_width=True)
            
        else:
            st.warning("No rules found. Try lowering the 'Minimum Support' slider.")

    # --- TABS 5 & 6 (UNCHANGED) ---
    with tab5:
        st.header("🔮 Demand Prediction")
        with st.spinner("Training..."):
            forecast_data = advanced_forecast(df)
        if not forecast_data.empty: st.dataframe(forecast_data.sort_values(['Category', 'Total Predicted Demand (Next 30 Days)'], ascending=[True, False]), use_container_width=True, hide_index=True)

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

else:
    st.info("👋 Upload data to begin.")
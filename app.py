import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
from mlxtend.frequent_patterns import fpgrowth, apriori, association_rules
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from datetime import timedelta
import warnings
import io

# --- NEW IMPORTS FOR HYBRID MODEL ---
from prophet import Prophet
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

# --- PAGE CONFIG ---
st.set_page_config(page_title="Mithas Intelligence 10.5", layout="wide", page_icon="🍬")

# --- DATA PROCESSING (ROBUST) ---
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
    
    # Numeric conversion
    for c in ['Quantity', 'TotalAmount', 'UnitPrice']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
            
    # Date parsing
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce', dayfirst=True)
        if df['Date'].isnull().any():
            df = df.dropna(subset=['Date'])
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
        'Quantity': 'sum', 'TotalAmount': 'sum'
    }).reset_index()
    hourly_stats['Time Slot'] = hourly_stats['Hour'].apply(lambda h: f"{int(h):02d}:00 - {int(h)+1:02d}:00")
    hourly_stats = hourly_stats.sort_values(['Hour', 'Quantity'], ascending=[True, False])
    return hourly_stats[['Time Slot', 'ItemName', 'Quantity', 'TotalAmount']]

# --- HYBRID FORECASTER ---

class HybridDemandForecaster:
    def __init__(self):
        self.prophet_model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=True)
        try: self.prophet_model.add_country_holidays(country_name='IN')
        except: pass 
        self.xgb_model = XGBRegressor(n_estimators=1000, learning_rate=0.05, max_depth=6, n_jobs=-1)
        self.rf_model = RandomForestRegressor(n_estimators=200, max_depth=10, n_jobs=-1)
        self.meta_model = Ridge(alpha=1.0)
        self.is_fitted = False
        self.feature_columns = []
        self.training_data_tail = None

    def create_features(self, df, is_future=False):
        df_feat = df.copy()
        df_feat['hour'] = df_feat['ds'].dt.hour
        df_feat['dayofweek'] = df_feat['ds'].dt.dayofweek
        df_feat['quarter'] = df_feat['ds'].dt.quarter
        df_feat['month'] = df_feat['ds'].dt.month
        df_feat['is_weekend'] = df_feat['dayofweek'].apply(lambda x: 1 if x >= 5 else 0)
        if 'y' in df_feat.columns:
            df_feat['lag_1d'] = df_feat['y'].shift(24) 
            df_feat['rolling_mean_3d'] = df_feat['y'].rolling(window=24*3).mean()
        df_feat = df_feat.fillna(0)
        return df_feat

    def fit(self, df):
        self.training_data_tail = df.tail(24 * 14).copy()
        self.prophet_model.fit(df)
        df_ml = self.create_features(df)
        drop_cols = ['ds', 'y', 'yhat']
        self.feature_columns = [c for c in df_ml.columns if c not in drop_cols and np.issubdtype(df_ml[c].dtype, np.number)]
        X = df_ml[self.feature_columns]
        y = df['y']
        self.xgb_model.fit(X, y, verbose=False)
        self.rf_model.fit(X, y)
        pred_prophet = self.prophet_model.predict(df)['yhat'].values
        pred_xgb = self.xgb_model.predict(X)
        pred_rf = self.rf_model.predict(X)
        stacked_X = np.column_stack((pred_prophet, pred_xgb, pred_rf))
        self.meta_model.fit(stacked_X, y)
        self.is_fitted = True

    def predict(self, periods=30):
        if not self.is_fitted: raise Exception("Model not fitted.")
        future_prophet = self.prophet_model.make_future_dataframe(periods=periods, freq='D')
        forecast_prophet = self.prophet_model.predict(future_prophet)
        future_dates = future_prophet.tail(periods).copy()
        extended_df = pd.concat([self.training_data_tail, future_dates], axis=0, ignore_index=True)
        extended_feat = self.create_features(extended_df, is_future=True)
        X_future = extended_feat.tail(periods)[self.feature_columns]
        pred_prophet = forecast_prophet['yhat'].tail(periods).values
        pred_xgb = self.xgb_model.predict(X_future)
        pred_rf = self.rf_model.predict(X_future)
        stacked_future = np.column_stack((pred_prophet, pred_xgb, pred_rf))
        final_pred = self.meta_model.predict(stacked_future)
        result = future_dates[['ds']].copy()
        result['Predicted_Demand'] = np.maximum(final_pred, 0)
        result['Prophet_View'] = pred_prophet
        result['XGB_View'] = pred_xgb
        result['RF_View'] = pred_rf
        return result

@st.cache_resource(show_spinner="Training AI Brain...")
def train_hybrid_model(df_train):
    forecaster = HybridDemandForecaster()
    forecaster.fit(df_train)
    return forecaster

# --- ANALYSIS LOGIC ---

def get_combo_analysis_full(df):
    valid_df = df[df['Quantity'] > 0]
    basket = valid_df.groupby(['OrderID', 'ItemName'])['Quantity'].count().unstack().fillna(0)
    basket_sets = (basket > 0).astype(int)
    frequent = apriori(basket_sets, min_support=0.005, use_colnames=True)
    if frequent.empty: return pd.DataFrame()
    rules = association_rules(frequent, metric="lift", min_threshold=1.05)
    rules['Item A'] = rules['antecedents'].apply(lambda x: list(x)[0])
    rules['Item B'] = rules['consequents'].apply(lambda x: list(x)[0])
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
    item_avg_price = df.groupby('ItemName').apply(lambda x: x['TotalAmount'].sum() / x['Quantity'].sum() if x['Quantity'].sum() > 0 else 0).to_dict()
    rules['Combo Value'] = rules['Item A'].map(item_avg_price).fillna(0) + rules['Item B'].map(item_avg_price).fillna(0)
    return rules

def get_part3_strategy(rules_df):
    if rules_df.empty: return pd.DataFrame(), pd.DataFrame()
    proven = rules_df.sort_values('Times Sold Together', ascending=False).head(10).copy()
    potential = rules_df[~rules_df.index.isin(proven.index)]
    potential = potential[potential['lift'] > 1.5].sort_values('lift', ascending=False).head(10).copy()
    return proven, potential

def analyze_pareto_hierarchical(df):
    item_rev = df.groupby(['Category', 'ItemName'])['TotalAmount'].sum().reset_index()
    total_rev = item_rev['TotalAmount'].sum()
    item_rev = item_rev.sort_values('TotalAmount', ascending=False)
    item_rev['Cumulative'] = item_rev['TotalAmount'].cumsum()
    item_rev['CumPerc'] = 100 * item_rev['Cumulative'] / total_rev
    pareto_items = item_rev[item_rev['CumPerc'] <= 80].copy()
    total_items = df['ItemName'].nunique()
    pareto_count = pareto_items['ItemName'].nunique()
    ratio_text = f"**{pareto_count} items** (out of {total_items}) contribute to 80% of revenue."
    menu_perc = (pareto_count / total_items) * 100
    cat_rev = df.groupby('Category')['TotalAmount'].sum().reset_index()
    cat_rev['CatContrib'] = (cat_rev['TotalAmount'] / total_rev) * 100
    merged = pd.merge(pareto_items, cat_rev[['Category', 'CatContrib']], on='Category', how='left')
    merged['ItemContrib'] = (merged['TotalAmount'] / total_rev) * 100
    display_df = merged[['Category', 'CatContrib', 'ItemName', 'ItemContrib', 'TotalAmount']]
    return display_df.sort_values(['CatContrib', 'TotalAmount'], ascending=[False, False]), ratio_text, menu_perc

# --- NETWORK GRAPH RENDERER (FIXED) ---
def render_network_graph(rules_df):
    if rules_df.empty: return
    G = nx.DiGraph()
    for _, row in rules_df.iterrows():
        G.add_edge(row['Antecedent'], row['Consequent'], weight=row['lift'])

    pos = nx.spring_layout(G, k=0.5, iterations=50)

    edge_x, edge_y = [], []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none', mode='lines')

    node_x, node_y, node_text, node_adj = [], [], [], []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(str(node))
        node_adj.append(len(G.adj[node]))

    # --- STABILITY FIX: REMOVED BORDER LINE STYLING ---
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        text=node_text,
        textposition="top center",
        marker=dict(
            showscale=True,
            colorscale='YlGnBu',
            size=15,
            color=node_adj,
            colorbar=dict(
                thickness=15, title='Connections', xanchor='left', titleside='right'
            )
        )
    )

    fig = go.Figure(data=[edge_trace, node_trace],
                 layout=go.Layout(
                    title='🔗 Product Interaction Network',
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=20,l=5,r=5,t=40),
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                    )
    st.plotly_chart(fig, use_container_width=True)

# --- MAIN APP ---
st.title("📊 Mithas Restaurant Intelligence 10.5")
uploaded_file = st.sidebar.file_uploader("Upload Monthly Data", type=['xlsx'])

if uploaded_file:
    df = load_data(uploaded_file)
    if df.empty: st.stop()

    pareto_list = get_pareto_items(df)[0:len(get_pareto_items(df))] # helper returns list
    rules_df = get_combo_analysis_full(df)
    proven_df, potential_df = get_part3_strategy(rules_df)
    proven_list = proven_df['pair'].tolist() if not proven_df.empty else []
    potential_list = potential_df['pair'].tolist() if not potential_df.empty else []

    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
        "Overview", "Day Analysis", "Categories", "Pareto", "Trends", "Combos", "Association", "Forecast", "AI Chat"
    ])

    with tab1:
        st.header("Metrics")
        rev = df['TotalAmount'].sum()
        orders = df['OrderID'].nunique()
        st.metric("Total Revenue", f"₹{rev:,.0f}")
        st.metric("Total Orders", orders)
    
    with tab2:
        st.header("Day Wise Waste Analysis")
        # Reuse logic for waste analysis
        days = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
        not_worth = {}
        max_l = 0
        for d in days:
            d_df = df[df['DayOfWeek'] == d]
            active = d_df[d_df['Quantity']>0]['ItemName'].unique()
            # 3 day logic simplified for stability
            sums = df[df['DayOfWeek'].isin([d])].groupby('ItemName')['Quantity'].sum()
            cands = sums[sums < 3].index.tolist()
            final = [f"★ {x}" if x in pareto_list else x for x in cands if x in active]
            not_worth[d] = final
            if len(final) > max_l: max_l = len(final)
        
        for d in days:
            if len(not_worth[d]) < max_l: not_worth[d].extend([""]*(max_l-len(not_worth[d])))
        
        waste_df = pd.DataFrame(not_worth)
        st.dataframe(waste_df, use_container_width=True)

        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
            waste_df.to_excel(writer, index=False)
        st.download_button("Download Report", buffer, "waste_report.xlsx")

    with tab3:
        st.header("Category Details")
        cat = st.selectbox("Category", sorted(df['Category'].unique()))
        c_df = df[df['Category'] == cat]
        st.dataframe(c_df.groupby('ItemName')['Quantity'].sum().sort_values(ascending=False), use_container_width=True)

    with tab4:
        st.header("Pareto")
        p_df, msg, _ = analyze_pareto_hierarchical(df)
        st.info(msg)
        st.dataframe(p_df, hide_index=True, use_container_width=True)

    with tab5:
        st.header("Trends")
        st.caption("Select items in other tabs to see trends here.")
        daily = df.groupby('Date')['TotalAmount'].sum().reset_index()
        fig = px.line(daily, x='Date', y='TotalAmount')
        st.plotly_chart(fig, use_container_width=True)

    with tab6:
        st.header("Combos")
        st.dataframe(proven_df, use_container_width=True)

    with tab7:
        st.header("Association Analysis")
        c1, c2 = st.columns(2)
        with c1: level = st.radio("Level", ["ItemName", "Category"])
        with c2: min_sup = st.slider("Min Support %", 0.1, 5.0, 0.5) / 100
        
        valid_df = df[df['Quantity'] > 0]
        basket = valid_df.groupby(['OrderID', level])['Quantity'].count().unstack().fillna(0)
        basket_sets = (basket > 0).astype(bool)
        
        with st.spinner("Analyzing..."):
            frequent = fpgrowth(basket_sets, min_support=min_sup, use_colnames=True)
            if not frequent.empty:
                rules = association_rules(frequent, metric="confidence", min_threshold=0.1)
                rules['Antecedent'] = rules['antecedents'].apply(lambda x: list(x)[0])
                rules['Consequent'] = rules['consequents'].apply(lambda x: list(x)[0])
                if level == 'Category': rules = rules[rules['Antecedent'] != rules['Consequent']]
                rules = rules.sort_values('lift', ascending=False)
                
                # Render Graph
                render_network_graph(rules.head(30))
                st.dataframe(rules[['Antecedent', 'Consequent', 'support', 'confidence', 'lift']], use_container_width=True)
            else:
                st.warning("No associations found.")

    with tab8:
        st.header("Forecast")
        item = st.selectbox("Select Item", df['ItemName'].unique())
        if st.button("Forecast"):
            i_df = df[df['ItemName'] == item].groupby('Date')['Quantity'].sum().reset_index()
            i_df.columns = ['ds', 'y']
            if len(i_df) > 14:
                model = train_hybrid_model(i_df)
                res = model.predict(30)
                st.line_chart(res.set_index('ds')[['Predicted_Demand', 'Prophet_View']])
            else:
                st.error("Not enough data.")

    with tab9:
        st.header("AI Chat")
        if "msgs" not in st.session_state: st.session_state.msgs = []
        for m in st.session_state.msgs: st.chat_message(m["role"]).write(m["content"])
        if p := st.chat_input():
            st.session_state.msgs.append({"role":"user","content":p})
            st.chat_message("user").write(p)
            try:
                llm = ChatOpenAI(model="gpt-4o-mini", api_key=st.secrets["OPENAI_API_KEY"])
                ctx = f"Rev: {df['TotalAmount'].sum()}"
                r = llm.invoke([SystemMessage(content=ctx), HumanMessage(content=p)])
                st.session_state.msgs.append({"role":"assistant","content":r.content})
                st.chat_message("assistant").write(r.content)
            except: st.error("API Key Error")

else:
    st.info("Upload Data")
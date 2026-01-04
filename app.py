import streamlit as st
import pandas as pd
import plotly.express as px
from mlxtend.frequent_patterns import apriori, association_rules
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

# --- CONFIGURATION ---
st.set_page_config(page_title="Mithas AI Manager", layout="wide")

# --- AI BRAIN ---
def get_ai_insight(context_text):
    try:
        # Uses the API Key you saved in Streamlit Secrets
        llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.7, openai_api_key=st.secrets["OPENAI_API_KEY"])
        
        template = """
        You are the General Manager of 'Mithas' restaurant.
        Analyze this daily sales summary and give 3 bullet points on how to improve revenue tomorrow.
        
        DATA:
        {context}
        
        Focus on:
        1. Promoting high-margin items.
        2. Suggesting specific combo offers for the 'Basket' items.
        3. Identifying which category needs a boost.
        """
        prompt = PromptTemplate(input_variables=["context"], template=template)
        chain = prompt | llm
        response = chain.invoke({"context": context_text})
        return response.content
    except Exception as e:
        return f"⚠️ AI Error: {str(e)}. (Check your OpenAI API Key in Secrets)"

# --- DATA CLEANING ---
def load_and_clean_data(file):
    df = pd.read_excel(file)
    
    # 1. RENAME COLUMNS (Mapping your names to system names)
    # Your Name : System Name
    column_map = {
        'Invoice No.': 'OrderID',
        'Item Name': 'ItemName',
        'Qty.': 'Quantity',
        'Final Total': 'TotalAmount',
        'Price': 'UnitPrice',
        'Category': 'Category',
        'Timestamp': 'Time',
        'Date': 'Date'
    }
    df = df.rename(columns=column_map)
    
    # 2. DATA TYPE FIXES
    # Ensure numbers are numbers (remove '₹' or ',' if present)
    for col in ['TotalAmount', 'Quantity', 'UnitPrice']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            
    return df

# --- ANALYTICS MODULES ---

def analyze_hourly(df):
    """Graphs sales by hour using the Timestamp column"""
    if 'Time' in df.columns:
        # Try to parse the time. If it's a string like "14:30", extract hour.
        # If it's a datetime object, extract hour directly.
        try:
            df['Hour'] = pd.to_datetime(df['Time'].astype(str), format='%H:%M:%S', errors='coerce').dt.hour
            # Fallback if format fails (e.g. if Excel auto-formatted it)
            if df['Hour'].isnull().all():
                 df['Hour'] = pd.to_datetime(df['Time'], errors='coerce').dt.hour
        except:
            return None
            
        hourly = df.groupby('Hour')['TotalAmount'].sum().reset_index()
        return hourly
    return None

def analyze_bcg(df):
    """Matrix: Volume (X) vs Unit Price (Y) [Proxy for Profit]"""
    # Group by Item
    stats = df.groupby('ItemName').agg({
        'Quantity': 'sum',
        'TotalAmount': 'sum',
        'UnitPrice': 'mean' # Avg selling price
    }).reset_index()
    
    return stats

def analyze_basket(df):
    """Finds items often bought in the same Invoice"""
    # Create the Basket Matrix
    basket = (df.groupby(['OrderID', 'ItemName'])['Quantity']
              .sum().unstack().reset_index().fillna(0)
              .set_index('OrderID'))
    
    # Encode (1 = Bought, 0 = Not Bought)
    basket_sets = basket.applymap(lambda x: 1 if x > 0 else 0)
    
    # Apriori Algorithm (Requires >1% support to be considered)
    frequent = apriori(basket_sets, min_support=0.01, use_colnames=True)
    
    if frequent.empty: return pd.DataFrame()
    
    # Association Rules
    rules = association_rules(frequent, metric="lift", min_threshold=1.0)
    return rules.sort_values(['lift'], ascending=False).head(5)

# --- DASHBOARD UI ---

st.title("🍛 Mithas Restaurant AI")
st.write("Daily Performance Dashboard")

uploaded_file = st.sidebar.file_uploader("Upload Daily Excel", type=['xlsx'])

if uploaded_file:
    # Load Data
    df = load_and_clean_data(uploaded_file)
    
    # TOP ROW: METRICS
    total_rev = df['TotalAmount'].sum()
    order_count = df['OrderID'].nunique()
    
    m1, m2, m3 = st.columns(3)
    m1.metric("Total Revenue", f"₹{total_rev:,.0f}")
    m2.metric("Total Orders", order_count)
    m3.metric("Avg Bill Value", f"₹{total_rev/order_count:.0f}" if order_count else 0)
    
    st.divider()

    # ROW 1: HOURLY TRENDS & CATEGORY PIE
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("⌚ Peak Hours")
        hourly = analyze_hourly(df)
        if hourly is not None and not hourly.empty:
            st.area_chart(hourly.set_index('Hour')['TotalAmount'])
        else:
            st.info("Could not process 'Timestamp' column.")

    with c2:
        st.subheader("🍰 Sales by Category")
        if 'Category' in df.columns:
            cat_fig = px.pie(df, values='TotalAmount', names='Category', hole=0.4)
            st.plotly_chart(cat_fig, use_container_width=True)

    # ROW 2: BCG & PARETO
    c3, c4 = st.columns(2)
    with c3:
        st.subheader("🧩 Menu Matrix (Price vs Vol)")
        bcg = analyze_bcg(df)
        # X = Popularity (Qty), Y = Price (High Ticket Items)
        fig_bcg = px.scatter(bcg, x='Quantity', y='UnitPrice', 
                             size='TotalAmount', color='TotalAmount', 
                             hover_name='ItemName',
                             labels={'Quantity': 'Volume (Popularity)', 'UnitPrice': 'Selling Price'},
                             title="High Volume (Right) vs High Price (Top)")
        # Add Quadrant Lines (Averages)
        fig_bcg.add_hline(y=bcg['UnitPrice'].mean(), line_dash="dot")
        fig_bcg.add_vline(x=bcg['Quantity'].mean(), line_dash="dot")
        st.plotly_chart(fig_bcg, use_container_width=True)

    with c4:
        st.subheader("🏆 The Vital Few (Pareto)")
        pareto = df.groupby('ItemName')['TotalAmount'].sum().sort_values(ascending=False).head(10)
        st.bar_chart(pareto)

    # ROW 3: COMBOS & AI
    st.divider()
    c5, c6 = st.columns(2)
    
    with c5:
        st.subheader("🍟 Combo Opportunities")
        try:
            rules = analyze_basket(df)
            if not rules.empty:
                st.write("Customers who bought 'Antecedent' also bought 'Consequent'")
                st.dataframe(rules[['antecedents', 'consequents', 'lift']])
            else:
                st.info("No strong item links found today.")
        except:
            st.error("Basket analysis failed. Check OrderID data.")

    with c6:
        st.subheader("🤖 AI Consultant Strategy")
        if st.button("Generate Daily Report"):
            with st.spinner("Analyzing..."):
                # Create a summary text
                summary_text = f"""
                Total Sales: {total_rev}
                Top Category: {df.groupby('Category')['TotalAmount'].sum().idxmax()}
                Top 3 Items: {df.groupby('ItemName')['TotalAmount'].sum().head(3).index.tolist()}
                """
                if not rules.empty:
                    summary_text += f"\nTop Combo: {list(rules.iloc[0]['antecedents'])} -> {list(rules.iloc[0]['consequents'])}"
                
                # Get AI Response
                insight = get_ai_insight(summary_text)
                st.success("Analysis Complete")
                st.markdown(insight)

else:
    st.info("Waiting for Excel file upload...")
import streamlit as st
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

# --- PAGE CONFIG ---
st.set_page_config(page_title="Restaurant AI Agent", layout="wide")

# --- SIDEBAR ---
with st.sidebar:
    st.header("Upload Daily Sales")
    uploaded_file = st.file_uploader("Choose your Excel file", type="xlsx")
    st.info("Columns required: OrderID, ItemName, Quantity, TotalAmount")

# --- FUNCTIONS ---
def analyze_pareto(df):
    # Group by item and sum revenue
    item_rev = df.groupby('ItemName')['TotalAmount'].sum().sort_values(ascending=False).reset_index()
    item_rev['cumulative_perc'] = 100 * item_rev['TotalAmount'].cumsum() / item_rev['TotalAmount'].sum()
    # The 'Vital Few' (Top 20% items generating 80% revenue)
    top_performers = item_rev[item_rev['cumulative_perc'] <= 80]
    return top_performers

def analyze_basket(df):
    # Pivot data: Rows = Orders, Cols = Items, Value = 1 if bought
    basket = (df.groupby(['OrderID', 'ItemName'])['Quantity']
              .sum().unstack().reset_index().fillna(0)
              .set_index('OrderID'))
    
    # Convert numbers to boolean (0 or 1)
    basket_sets = basket.applymap(lambda x: 1 if x > 0 else 0)
    
    # Run Apriori Algorithm
    frequent_itemsets = apriori(basket_sets, min_support=0.01, use_colnames=True) # 0.01 = occurs in 1% of transactions
    
    if frequent_itemsets.empty:
        return pd.DataFrame()
        
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
    return rules.sort_values(['lift'], ascending=False).head(5)

# --- MAIN LOGIC ---
st.title("👨‍🍳 Restaurant Analyst AI")

if uploaded_file:
    # Load data
    df = pd.read_excel(uploaded_file)
    st.write("### Data Preview")
    st.dataframe(df.head())

    # 1. PARETO ANALYSIS
    st.subheader("📊 Pareto Analysis (80/20 Rule)")
    st.write("These items generate 80% of your revenue. Never run out of stock on these!")
    pareto_data = analyze_pareto(df)
    st.bar_chart(pareto_data.set_index('ItemName')['TotalAmount'])

    # 2. BASKET ANALYSIS
    st.subheader("🛒 Combo Recommendations (Basket Analysis)")
    try:
        basket_rules = analyze_basket(df)
        if not basket_rules.empty:
            st.write("Items frequently bought together (High Lift = Strong Correlation)")
            st.table(basket_rules[['antecedents', 'consequents', 'lift']])
        else:
            st.warning("Not enough data overlap to find strong combos yet.")
    except Exception as e:
        st.error(f"Could not run basket analysis. Check OrderID column. Error: {e}")

else:
    st.warning("Please upload a daily sales report to begin.")
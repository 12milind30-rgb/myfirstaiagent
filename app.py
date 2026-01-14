import os
import warnings
from datetime import timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import streamlit as st
from mlxtend.frequent_patterns import apriori, association_rules, fpgrowth
from mlxtend.preprocessing import TransactionEncoder
from prophet import Prophet
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

# Optional LLM chat integration imports (may not be available in all envs)
try:
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage, SystemMessage
    LLM_AVAILABLE = True
except Exception:
    LLM_AVAILABLE = False

warnings.filterwarnings("ignore")

st.set_page_config(page_title="Mithas Intelligence 10.2", layout="wide")


# ----------------------
# Data loading & cleaning
# ----------------------
@st.cache_data
def load_data(file):
    """Load and preprocess data with robust type handling."""
    try:
        df = pd.read_excel(file)
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return None

    # Normalize columns if present
    col_map = {
        "Invoice No.": "OrderID",
        "Invoice No. ": "OrderID",
        "Item Name": "ItemName",
        "Qty.": "Quantity",
        "Qty. ": "Quantity",
        "Final Total": "TotalAmount",
        "Price": "UnitPrice",
        "Category": "Category",
        "Timestamp": "Time",
        "Date": "Date",
    }
    df = df.rename(columns={c: col_map[c] for c in df.columns if c in col_map})

    # Ensure numeric columns
    for c in ["Quantity", "TotalAmount", "UnitPrice"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    # Parse Date
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"])
        df["DayOfWeek"] = df["Date"].dt.day_name()

    # Extract Hour from Time if possible
    if "Time" in df.columns:
        try:
            df["Hour"] = pd.to_datetime(
                df["Time"].astype(str), format="%H:%M:%S", errors="coerce"
            ).dt.hour
            if df["Hour"].isnull().all():
                df["Hour"] = pd.to_datetime(df["Time"], errors="coerce").dt.hour
            df["Hour"] = df["Hour"].fillna(0).astype(int)
        except Exception:
            df["Hour"] = 0
    else:
        # If no Time column, but Date exists, set Hour to 0
        if "Date" in df.columns:
            df["Hour"] = 0

    # Make OrderID string to avoid numeric aggregation mismatches
    if "OrderID" in df.columns:
        df["OrderID"] = df["OrderID"].astype(str)

    return df


# ----------------------
# Helper utilities
# ----------------------
def get_pareto_items(df):
    """Return items that cumulatively make up < 80% revenue."""
    item_rev = df.groupby("ItemName")["TotalAmount"].sum().sort_values(ascending=False).reset_index()
    total_revenue = item_rev["TotalAmount"].sum()
    if total_revenue == 0:
        return []
    item_rev["Cumulative"] = item_rev["TotalAmount"].cumsum()
    item_rev["CumPerc"] = 100 * item_rev["Cumulative"] / total_revenue
    return item_rev[item_rev["CumPerc"] < 80]["ItemName"].tolist()


def get_peak_hour_for_pair(df, item_a, item_b):
    """Return peak hour slot for orders that contain both item_a and item_b."""
    try:
        orders_a = set(df[df["ItemName"] == item_a]["OrderID"].dropna())
        orders_b = set(df[df["ItemName"] == item_b]["OrderID"].dropna())
        common_orders = list(orders_a.intersection(orders_b))
        if not common_orders:
            return "N/A"
        subset = df[df["OrderID"].isin(common_orders)]
        if "Hour" not in subset.columns or subset.empty:
            return "N/A"
        peak = subset["Hour"].mode()
        if not peak.empty:
            p = int(peak.iloc[0])
            return f"{p:02d}:00 - {p+1:02d}:00"
        return "N/A"
    except Exception:
        return "N/A"


def get_hourly_details(df):
    """Return aggregated hourly stats (9-23 hours)."""
    if "Hour" not in df.columns:
        return pd.DataFrame()
    mask = (df["Hour"] >= 9) & (df["Hour"] <= 23)
    filtered = df[mask].copy()
    if filtered.empty:
        return pd.DataFrame()
    hourly_stats = (
        filtered.groupby(["Hour", "ItemName"])
        .agg({"Quantity": "sum", "TotalAmount": "sum"})
        .reset_index()
    )
    hourly_stats["Time Slot"] = hourly_stats["Hour"].apply(lambda h: f"{int(h):02d}:00 - {int(h)+1:02d}:00")
    hourly_stats = hourly_stats.sort_values(["Hour", "Quantity"], ascending=[True, False])
    return hourly_stats[["Time Slot", "ItemName", "Quantity", "TotalAmount"]]


# ----------------------
# Hybrid Forecast Model
# ----------------------
class HybridDemandForecaster:
    def __init__(self, seasonality_mode="multiplicative"):
        self.prophet_model = Prophet(
            seasonality_mode=seasonality_mode,
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=True,
            interval_width=0.95,
        )
        try:
            self.prophet_model.add_country_holidays(country_name="IN")
        except Exception:
            pass
        self.xgb_model = XGBRegressor(
            n_estimators=1000,
            learning_rate=0.05,
            max_depth=6,
            n_jobs=-1,
            objective="reg:squarederror",
            random_state=42,
            verbosity=0,
        )
        self.rf_model = RandomForestRegressor(n_estimators=200, max_depth=10, n_jobs=-1, random_state=42)
        self.meta_model = Ridge(alpha=1.0)
        self.scaler = StandardScaler()
        self.last_training_date = None
        self.training_data_tail = None
        self.is_fitted = False
        self.feature_columns = []

    def create_features(self, df, is_future=False):
        df_feat = df.copy()
        df_feat["hour"] = df_feat["ds"].dt.hour
        df_feat["dayofweek"] = df_feat["ds"].dt.dayofweek
        df_feat["quarter"] = df_feat["ds"].dt.quarter
        df_feat["month"] = df_feat["ds"].dt.month
        df_feat["is_weekend"] = df_feat["dayofweek"].apply(lambda x: 1 if x >= 5 else 0)
        if "y" in df_feat.columns:
            df_feat["lag_1d"] = df_feat["y"].shift(1)
            df_feat["lag_7d"] = df_feat["y"].shift(7)
            df_feat["rolling_mean_3d"] = df_feat["y"].rolling(window=3).mean()
            df_feat["rolling_std_7d"] = df_feat["y"].rolling(window=7).std()
        df_feat = df_feat.fillna(0)
        return df_feat

    def fit(self, df):
        if len(df) < 14:
            raise ValueError("Need at least 14 days of data to train")
        self.last_training_date = df["ds"].max()
        # keep recent tail for features when predicting
        self.training_data_tail = df.tail(14).copy()
        self.prophet_model.fit(df)
        df_ml = self.create_features(df)
        drop_cols = ["ds", "y", "yhat"]
        self.feature_columns = [c for c in df_ml.columns if c not in drop_cols and np.issubdtype(df_ml[c].dtype, np.number)]
        X = df_ml[self.feature_columns]
        y = df["y"]
        if X.empty:
            raise ValueError("No features available for training")
        self.xgb_model.fit(X, y)
        self.rf_model.fit(X, y)
        pred_prophet = self.prophet_model.predict(df)["yhat"].values
        pred_xgb = self.xgb_model.predict(X)
        pred_rf = self.rf_model.predict(X)
        stacked_X = np.column_stack((pred_prophet, pred_xgb, pred_rf))
        self.meta_model.fit(stacked_X, y)
        self.is_fitted = True

    def predict(self, periods=30):
        if not self.is_fitted:
            raise Exception("Model not fitted yet.")
        future_prophet = self.prophet_model.make_future_dataframe(periods=periods, freq="D")
        forecast_prophet = self.prophet_model.predict(future_prophet)
        future_dates = future_prophet.tail(periods).copy().reset_index(drop=True)
        # prepare extended series for feature creation
        tail = self.training_data_tail.copy().rename(columns={"ds": "ds", "y": "y"})
        # combine tail (as recent history) and future_dates (no y)
        combined = pd.concat([tail[["ds", "y"]], future_dates[["ds"]].assign(y=0)], ignore_index=True)
        combined = combined.reset_index(drop=True)
        feat = self.create_features(combined, is_future=True)
        X_future = feat.tail(periods)[self.feature_columns]
        pred_prophet = forecast_prophet["yhat"].tail(periods).values
        pred_xgb = self.xgb_model.predict(X_future)
        pred_rf = self.rf_model.predict(X_future)
        stacked_future = np.column_stack((pred_prophet, pred_xgb, pred_rf))
        final_pred = self.meta_model.predict(stacked_future)
        result = future_dates[["ds"]].copy()
        result["Predicted_Demand"] = np.maximum(final_pred, 0)
        result["Prophet_View"] = pred_prophet
        result["XGB_View"] = pred_xgb
        result["RF_View"] = pred_rf
        return result


# ----------------------
# Improved Association Analyzer
# ----------------------
class ImprovedAssociationAnalyzer:
    """Enhanced association analysis with reliability metrics."""

    def __init__(self, min_support=0.005, min_confidence=0.1, min_conviction=1.0):
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.min_conviction = min_conviction

    @staticmethod
    def calculate_all_metrics(rules):
        if rules.empty:
            return rules
        # Leverage
        rules["leverage"] = rules["support"] - (rules["antecedent support"] * rules["consequent support"])
        # Jaccard
        denom = (rules["antecedent support"] + rules["consequent support"] - rules["support"]).replace(0, np.nan)
        rules["jaccard"] = (rules["support"] / denom).fillna(0)
        # Kulczynski
        rules["kulczynski"] = 0.5 * (rules["confidence"] + (rules["support"] / rules["consequent support"]).replace(np.inf, 0).fillna(0))
        # Certainty factor
        denom_cf = (1 - rules["consequent support"]).replace(0, np.nan)
        rules["certainty_factor"] = ((rules["confidence"] - rules["consequent support"]) / denom_cf).replace([np.inf, -np.inf], 0).fillna(0)
        return rules

    @staticmethod
    def rank_by_reliability(rules):
        if rules.empty:
            return rules
        # safe normalization
        rules = rules.copy()
        rules["norm_lift"] = (rules["lift"] - 1) / (rules["lift"].max() - 1 + 1e-9)
        rules["norm_conviction"] = np.minimum(rules["conviction"] / (rules["conviction"].max() + 1e-9), 1.0)
        rules["norm_leverage"] = (rules["leverage"] - rules["leverage"].min()) / (rules["leverage"].max() - rules["leverage"].min() + 1e-9)
        rules["reliability_score"] = 0.4 * rules["norm_lift"] + 0.3 * rules["norm_conviction"] + 0.3 * rules["norm_leverage"]
        return rules.sort_values("reliability_score", ascending=False)


# ----------------------
# Association utilities
# ----------------------
def run_advanced_association(df, level="ItemName", min_sup=0.005, min_conf=0.1):
    """Run association rules and return cleaned DataFrame with key metrics."""
    valid_df = df[df["Quantity"] > 0].copy()
    if valid_df.empty:
        return pd.DataFrame()
    qty_basket = valid_df.groupby(["OrderID", level])["Quantity"].sum().unstack().fillna(0)
    basket_sets = (qty_basket > 0).astype(bool)
    frequent_itemsets = fpgrowth(basket_sets, min_support=min_sup, use_colnames=True)
    if frequent_itemsets.empty:
        return pd.DataFrame()
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_conf)
    if rules.empty:
        return pd.DataFrame()
    rules["Support (%)"] = rules["support"] * 100
    rules["Antecedent"] = rules["antecedents"].apply(lambda x: list(x)[0] if len(x) > 0 else "")
    rules["Consequent"] = rules["consequents"].apply(lambda x: list(x)[0] if len(x) > 0 else "")
    if level == "Category":
        rules = rules[rules["Antecedent"] != rules["Consequent"]]
    # Total item qty on common orders
    def calculate_split_quantity(row):
        ant = row["Antecedent"]
        con = row["Consequent"]
        if ant not in qty_basket.columns or con not in qty_basket.columns:
            return "0 (0 + 0)"
        common_orders_mask = (qty_basket[ant] > 0) & (qty_basket[con] > 0)
        sum_ant = int(qty_basket.loc[common_orders_mask, ant].sum())
        sum_con = int(qty_basket.loc[common_orders_mask, con].sum())
        total = sum_ant + sum_con
        return f"{total} ({sum_ant} + {sum_con})"
    rules["Total Item Qty"] = rules.apply(calculate_split_quantity, axis=1)
    # Zhang metric
    def zhangs_metric(rule):
        sup = rule["support"]
        sup_a = rule["antecedent support"]
        sup_c = rule["consequent support"]
        numerator = sup - (sup_a * sup_c)
        denominator = max(sup * (1 - sup_a), sup_a * (sup_c - sup))
        if denominator == 0:
            return 0
        return numerator / denominator
    rules["zhang"] = rules.apply(zhangs_metric, axis=1)
    rules = rules.sort_values("lift", ascending=False)
    rules["pair_key"] = rules.apply(lambda x: frozenset([x["Antecedent"], x["Consequent"]]), axis=1)
    rules = rules.drop_duplicates(subset=["pair_key"])
    keep_cols = ["Antecedent", "Consequent", "Support (%)", "Total Item Qty", "confidence", "lift", "zhang", "conviction"]
    return rules[keep_cols].reset_index(drop=True)


def get_combo_analysis_full(df):
    """Return item-item combos using apriori and lift."""
    valid_df = df[df["Quantity"] > 0].copy()
    if valid_df.empty:
        return pd.DataFrame()
    basket = valid_df.groupby(["OrderID", "ItemName"])["Quantity"].count().unstack().fillna(0)
    basket_sets = (basket > 0).astype(int)
    frequent = apriori(basket_sets, min_support=0.005, use_colnames=True)
    if frequent.empty:
        return pd.DataFrame()
    rules = association_rules(frequent, metric="lift", min_threshold=1.05)
    if rules.empty:
        return pd.DataFrame()
    rules["Item A"] = rules["antecedents"].apply(lambda x: list(x)[0] if len(x) > 0 else "")
    rules["Item B"] = rules["consequents"].apply(lambda x: list(x)[0] if len(x) > 0 else "")
    rules = rules[rules["Item A"] != rules["Item B"]]
    rules["pair"] = rules.apply(lambda x: tuple(sorted([x["Item A"], x["Item B"]])), axis=1)
    rules = rules.drop_duplicates(subset=["pair"])
    item_cat_map = df.set_index("ItemName")["Category"].to_dict() if "ItemName" in df.columns else {}
    rules["Category A"] = rules["Item A"].map(item_cat_map)
    rules["Category B"] = rules["Item B"].map(item_cat_map)
    rules["Specific Item Combo"] = rules["Item A"] + " + " + rules["Item B"]
    total_transactions = len(basket_sets)
    rules["Times Sold Together"] = (rules["support"] * total_transactions).round().astype(int).clip(lower=1)
    rules["Peak Hour"] = rules.apply(lambda x: get_peak_hour_for_pair(df, x["Item A"], x["Item B"]), axis=1)
    item_avg_price = df.groupby("ItemName").apply(lambda x: x["TotalAmount"].sum() / x["Quantity"].sum() if x["Quantity"].sum() > 0 else 0).to_dict()
    rules["Combo Value"] = rules["Item A"].map(item_avg_price).fillna(0) + rules["Item B"].map(item_avg_price).fillna(0)
    return rules.reset_index(drop=True)


def get_part3_strategy(rules_df):
    if rules_df.empty:
        return pd.DataFrame(), pd.DataFrame()
    proven = rules_df.sort_values("Times Sold Together", ascending=False).head(10).copy()
    potential = rules_df.loc[~rules_df.index.isin(proven.index)]
    potential = potential[potential["lift"] > 1.5].sort_values("lift", ascending=False).head(10).copy()
    return proven, potential


# ----------------------
# Analytics functions
# ----------------------
def get_overview_metrics(df):
    total_rev = df["TotalAmount"].sum()
    total_orders = df["OrderID"].nunique() if "OrderID" in df.columns else 0
    num_days = df["Date"].nunique() if "Date" in df.columns else 0
    avg_rev_day = total_rev / num_days if num_days > 0 else 0
    num_weeks = max(1, num_days / 7) if num_days > 0 else 1
    avg_rev_week = total_rev / num_weeks
    aov = total_rev / total_orders if total_orders > 0 else 0
    return total_rev, total_orders, avg_rev_day, avg_rev_week, aov


def get_star_items_with_hours(df, limit_n):
    total_rev = df["TotalAmount"].sum()
    if total_rev == 0:
        return pd.DataFrame(columns=["ItemName", "TotalAmount", "Contribution %", "Peak Selling Hour", "Qty Sold (Peak)"])
    item_stats = df.groupby("ItemName").agg({"TotalAmount": "sum"}).reset_index()
    item_stats["Contribution %"] = (item_stats["TotalAmount"] / total_rev) * 100
    item_stats = item_stats.sort_values("TotalAmount", ascending=False).head(limit_n)
    peak_hours = []
    peak_qtys = []
    for item in item_stats["ItemName"]:
        item_data = df[df["ItemName"] == item]
        if "Hour" in df.columns and not item_data.empty:
            hour_grouped = item_data.groupby("Hour")["Quantity"].sum()
            if not hour_grouped.empty:
                peak_hour = int(hour_grouped.idxmax())
                peak_hours.append(f"{peak_hour:02d}:00 - {peak_hour+1:02d}:00")
                peak_qtys.append(int(hour_grouped.max()))
            else:
                peak_hours.append("N/A")
                peak_qtys.append(0)
        else:
            peak_hours.append("N/A")
            peak_qtys.append(0)
    item_stats["Peak Selling Hour"] = peak_hours
    item_stats["Qty Sold (Peak)"] = peak_qtys
    return item_stats


def analyze_peak_hour_items(df):
    if "Hour" not in df.columns:
        return pd.DataFrame(), []
    hourly_rev = df.groupby("Hour")["TotalAmount"].sum()
    if hourly_rev.empty:
        return pd.DataFrame(), []
    top_3_hours = hourly_rev.nlargest(3).index.tolist()
    peak_df = df[df["Hour"].isin(top_3_hours)]
    if peak_df.empty:
        return pd.DataFrame(), top_3_hours
    top_items = peak_df.groupby("ItemName")["Quantity"].sum().sort_values(ascending=False).head(10).reset_index()
    top_items.columns = ["Item Name", "Qty Sold in Peak Hours"]
    return top_items, top_3_hours


def get_contribution_lists(df):
    total_rev = df["TotalAmount"].sum()
    if total_rev == 0:
        return pd.DataFrame(columns=["Category", "TotalAmount", "Contribution"])
    cat_df = df.groupby("Category")["TotalAmount"].sum().reset_index()
    cat_df["Contribution"] = (cat_df["TotalAmount"] / total_rev) * 100
    cat_df = cat_df.sort_values("TotalAmount", ascending=False)
    return cat_df


def analyze_pareto_hierarchical(df):
    item_rev = df.groupby(["Category", "ItemName"])["TotalAmount"].sum().reset_index()
    total_revenue = item_rev["TotalAmount"].sum()
    if total_revenue == 0:
        return pd.DataFrame(), "", 0
    item_rev = item_rev.sort_values("TotalAmount", ascending=False)
    item_rev["Cumulative"] = item_rev["TotalAmount"].cumsum()
    item_rev["CumPerc"] = 100 * item_rev["Cumulative"] / total_revenue
    pareto_items = item_rev[item_rev["CumPerc"] < 80].copy()
    total_unique_items = df["ItemName"].nunique()
    pareto_unique_items = pareto_items["ItemName"].nunique()
    ratio_text = f"**{pareto_unique_items} items** (out of {total_unique_items}) contribute to 80% of your revenue."
    percentage_of_menu = (pareto_unique_items / total_unique_items) * 100 if total_unique_items > 0 else 0
    cat_rev = df.groupby("Category")["TotalAmount"].sum().reset_index()
    cat_rev["CatContrib"] = (cat_rev["TotalAmount"] / total_revenue) * 100
    merged = pd.merge(pareto_items, cat_rev[["Category", "CatContrib"]], on="Category", how="left")
    merged["ItemContrib"] = (merged["TotalAmount"] / total_revenue) * 100
    display_df = merged[["Category", "CatContrib", "ItemName", "ItemContrib", "TotalAmount"]]
    display_df = display_df.sort_values(["CatContrib", "TotalAmount"], ascending=[False, False])
    return display_df, ratio_text, percentage_of_menu


def plot_time_series_fixed(df, pareto_list, n_items):
    categories = df["Category"].unique() if "Category" in df.columns else []
    for cat in categories:
        st.subheader(f"📈 {cat}")
        cat_data = df[df["Category"] == cat]
        if cat_data.empty:
            st.warning(f"No data for {cat}")
            continue
        top_items = cat_data.groupby("ItemName")["TotalAmount"].sum().sort_values(ascending=False).head(n_items).index.tolist()
        subset = cat_data[cat_data["ItemName"].isin(top_items)]
        if subset.empty:
            continue
        daily = subset.groupby(["Date", "ItemName"])["Quantity"].sum().reset_index()
        daily["Legend Name"] = daily["ItemName"].apply(lambda x: f"★ {x}" if x in pareto_list else x)
        fig = px.line(daily, x="Date", y="Quantity", color="Legend Name", markers=True)
        for item in top_items:
            item_daily = daily[daily["ItemName"] == item]
            if not item_daily.empty:
                avg_val = item_daily["Quantity"].mean()
                fig.add_hline(y=avg_val, line_dash="dot", line_color="grey", opacity=0.5)
                fig.add_annotation(
                    x=daily["Date"].max(),
                    y=avg_val,
                    text=f"{item}: {avg_val:.1f}",
                    showarrow=False,
                    yshift=10,
                    font=dict(color="red", size=10),
                )
        fig.update_xaxes(dtick="D2", tickformat="%d %b (%a)")
        fig.update_yaxes(matches=None, showticklabels=True)
        st.plotly_chart(fig, use_container_width=True)


# ----------------------
# Main app
# ----------------------
st.title("📊 Mithas Restaurant Intelligence 10.2")
uploaded_file = st.sidebar.file_uploader("Upload Monthly Data (Sidebar)", type=["xlsx"])

if uploaded_file:
    df = load_data(uploaded_file)
    if df is None or df.empty:
        st.error("Failed to load data or file is empty")
        st.stop()

    pareto_list = get_pareto_items(df)
    pareto_count = len(pareto_list)

    rules_df = get_combo_analysis_full(df)
    proven_df, potential_df = get_part3_strategy(rules_df)
    proven_list = proven_df["pair"].tolist() if not proven_df.empty else []
    potential_list = potential_df["pair"].tolist() if not potential_df.empty else []

    tabs = st.tabs(
        [
            "Overview",
            "Day wise Analysis",
            "Category Details",
            "Pareto (Visual)",
            "Time Series",
            "Smart Combos",
            "Association Analysis",
            "Demand Forecast",
            "AI Chat",
        ]
    )
    tab1, tab_day_wise, tab_cat, tab2, tab3, tab4, tab_assoc, tab5, tab6 = tabs

    # --- Overview tab ---
    with tab1:
        st.header("🏢 Business Overview")
        with st.expander("🛠️ Reorder Page Layout", expanded=False):
            overview_order = st.multiselect(
                "Select order of sections:",
                [
                    "Metrics",
                    "Graphs",
                    "Hourly Breakdown (Aggregated)",
                    "Peak Items List",
                    "Contributions",
                    "Star Items",
                ],
                default=[
                    "Metrics",
                    "Graphs",
                    "Hourly Breakdown (Aggregated)",
                    "Peak Items List",
                    "Contributions",
                    "Star Items",
                ],
            )

        def render_metrics():
            rev, orders, avg_day, avg_week, aov = get_overview_metrics(df)
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("💰 Total Revenue", f"₹{rev:,.0f}")
            c2.metric("🧾 Total Orders", orders)
            c3.metric("📅 Avg Rev/Day", f"₹{avg_day:,.0f}")
            c4.metric("🗓️ Avg Rev/Week", f"₹{avg_week:,.0f}")
            c5.metric("💳 Avg Order Value", f"₹{aov:,.0f}")
            st.divider()

        def render_graphs():
            g1, g2 = st.columns(2)
            with g1:
                st.subheader("⌚ Peak Hours Graph")
                if "Hour" in df.columns:
                    hourly = df.groupby("Hour")["TotalAmount"].sum().reset_index()
                    if not hourly.empty:
                        fig_hourly = px.bar(hourly, x="Hour", y="TotalAmount")
                        avg_hourly = hourly["TotalAmount"].mean()
                        fig_hourly.add_hline(
                            y=avg_hourly,
                            line_dash="dash",
                            line_color="red",
                            annotation_text=f"Avg: ₹{avg_hourly:,.0f}",
                            annotation_position="top right",
                        )
                        fig_hourly.update_xaxes(tickmode="linear", dtick=1)
                        st.plotly_chart(fig_hourly, use_container_width=True)
            with g2:
                st.subheader("📅 Peak Days Graph")
                if "DayOfWeek" in df.columns:
                    daily_peak = (
                        df.groupby("DayOfWeek")["TotalAmount"]
                        .sum()
                        .reindex(
                            [
                                "Monday",
                                "Tuesday",
                                "Wednesday",
                                "Thursday",
                                "Friday",
                                "Saturday",
                                "Sunday",
                            ]
                        )
                        .reset_index()
                    )
                    daily_peak.columns = ["DayOfWeek", "TotalAmount"]
                    fig_daily = px.bar(daily_peak, x="DayOfWeek", y="TotalAmount")
                    fig_daily.update_xaxes(tickfont=dict(family="Arial Black", size=14, color="black"))
                    st.plotly_chart(fig_daily, use_container_width=True)
            st.divider()

        def render_hourly_aggregated():
            st.subheader("🕰️ Hourly Sales Breakdown (9am - 11pm) - Aggregated")
            st.caption("Aggregated view of all days combined.")
            hourly_df = get_hourly_details(df)
            if not hourly_df.empty:
                hourly_df["ItemName"] = hourly_df["ItemName"].apply(lambda x: f"★ {x}" if x in pareto_list else x)
                time_slots = hourly_df["Time Slot"].unique()
                for slot in time_slots:
                    slot_data = hourly_df[hourly_df["Time Slot"] == slot]
                    total_rev = slot_data["TotalAmount"].sum()
                    total_qty = slot_data["Quantity"].sum()
                    top_item = slot_data.sort_values("Quantity", ascending=False).iloc[0]["ItemName"]
                    with st.expander(
                        f"⏰ {slot}  |  Revenue: ₹{total_rev:,.0f}  |  Units: {total_qty}  |  Top:  {top_item}"
                    ):
                        st.dataframe(slot_data[["ItemName", "Quantity", "TotalAmount"]], hide_index=True, use_container_width=True)
            else:
                st.info("No hourly data available.")
            st.divider()

        def render_peak_list():
            l1, l2 = st.columns(2)
            with l1:
                peak_items_df, top_hrs = analyze_peak_hour_items(df)
                st.subheader(f"🔥 Items Sold in Peak Hours {top_hrs}")
                if not peak_items_df.empty:
                    st.dataframe(peak_items_df, hide_index=True, use_container_width=True)
                else:
                    st.info("No peak hour data.")
            with l2:
                st.subheader("💰 High Revenue Days")
                top_days = df.groupby("Date")["TotalAmount"].sum().sort_values(ascending=False).head(5).reset_index()
                top_days["Date"] = top_days["Date"].dt.strftime("%Y-%m-%d (%A)")
                st.dataframe(top_days, hide_index=True, use_container_width=True)
            st.divider()

        def render_contributions():
            cat_cont = get_contribution_lists(df)
            st.subheader("📂 Category Contribution")
            if not cat_cont.empty:
                fig_pie = px.pie(cat_cont, values="TotalAmount", names="Category", hole=0.3)
                st.plotly_chart(fig_pie, use_container_width=True)
            else:
                st.info("No category contribution data.")
            st.divider()

        def render_star_items():
            st.subheader("⭐ Top Star Items & Selling Hours")
            slider_max = max(10, pareto_count)
            n_star = st.slider("Select Number of Star Items", 10, slider_max, 20)
            star_df = get_star_items_with_hours(df, n_star)
            if not star_df.empty:
                star_df["Item Name"] = star_df["ItemName"].apply(lambda x: f"★ {x}" if x in pareto_list else x)
                st.dataframe(
                    star_df[["Item Name", "TotalAmount", "Contribution %", "Peak Selling Hour", "Qty Sold (Peak)"]],
                    hide_index=True,
                    use_container_width=True,
                )
            else:
                st.info("No star items data.")

        block_map = {
            "Metrics": render_metrics,
            "Graphs": render_graphs,
            "Hourly Breakdown (Aggregated)": render_hourly_aggregated,
            "Peak Items List": render_peak_list,
            "Contributions": render_contributions,
            "Star Items": render_star_items,
        }

        for block_name in overview_order:
            if block_name in block_map:
                block_map[block_name]()

    # --- Day wise analysis ---
    with tab_day_wise:
        st.header("📅 Day-wise Deep Dive")
        st.subheader("⚠️ Items Not Worth Producing")
        st.markdown("Items with **< 3 units sold** in a 3-Day window. Only items with >0 sales on the specific day are shown.")
        days_list = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        available_categories = sorted(df["Category"].unique().tolist()) if "Category" in df.columns else []
        dropdown_options = ["All Categories"] + available_categories
        selected_nw_category = st.selectbox("Filter by Category", options=dropdown_options, index=0, key="nw_category_dropdown")
        not_worth_dict = {}
        max_len = 0
        for d in days_list:
            curr_idx = days_list.index(d)
            prev_d = days_list[(curr_idx - 1) % 7]
            next_d = days_list[(curr_idx + 1) % 7]
            window_days = [prev_d, d, next_d]
            window_df = df[df["DayOfWeek"].isin(window_days)].copy() if "DayOfWeek" in df.columns else pd.DataFrame()
            if selected_nw_category != "All Categories" and not window_df.empty:
                window_df = window_df[window_df["Category"] == selected_nw_category]
            item_sums = window_df.groupby("ItemName")["Quantity"].sum() if not window_df.empty else pd.Series(dtype=float)
            candidate_items = item_sums[item_sums < 3].index.tolist() if not item_sums.empty else []
            day_specific_df = df[df["DayOfWeek"] == d].copy() if "DayOfWeek" in df.columns else pd.DataFrame()
            if selected_nw_category != "All Categories" and not day_specific_df.empty:
                day_specific_df = day_specific_df[day_specific_df["Category"] == selected_nw_category]
            active_items_today = day_specific_df[day_specific_df["Quantity"] > 0]["ItemName"].unique().tolist() if not day_specific_df.empty else []
            final_items = [item for item in candidate_items if item in active_items_today]
            final_items = [f"★ {x}" if x in pareto_list else x for x in final_items]
            not_worth_dict[d] = final_items
            if len(final_items) > max_len:
                max_len = len(final_items)
        for d in days_list:
            current_len = len(not_worth_dict.get(d, []))
            if current_len < max_len:
                not_worth_dict[d].extend([""] * (max_len - current_len))
        waste_df = pd.DataFrame(not_worth_dict)
        st.dataframe(waste_df, use_container_width=True, height=500)
        st.markdown("---")
        st.subheader("📉 Missed Upsell Opportunities (Orders Below AOV)")
        monthly_aov = df["TotalAmount"].sum() / df["OrderID"].nunique() if "OrderID" in df.columns else 0
        if "Date" in df.columns:
            daily_aov_stats = (
                df.groupby(df["Date"].dt.date)
                .apply(
                    lambda x: pd.Series(
                        {
                            "Total Orders": len(x),
                            "Below AOV": (x["TotalAmount"] < monthly_aov).sum(),
                            "Percentage Below AOV": ((x["TotalAmount"] < monthly_aov).sum() / len(x)) * 100 if len(x) > 0 else 0,
                        }
                    )
                )
                .reset_index()
                .rename(columns={"Date": "Day"})
            )
            fig_upsell = px.bar(
                daily_aov_stats,
                x="Date",
                y="Percentage Below AOV",
                title=f"Daily % of Orders Below Monthly Average (₹{monthly_aov:,.0f})",
                color="Percentage Below AOV",
                color_continuous_scale="RdYlGn_r",
                labels={"Percentage Below AOV": "% Low Value Orders"},
                hover_data=["Total Orders", "Below AOV"],
            )
            fig_upsell.add_hline(y=50, line_dash="dot", line_color="red", annotation_text="Critical Threshold (50%)", annotation_position="top right")
            fig_upsell.update_layout(xaxis_title=None, yaxis_title="% of Orders < AOV", coloraxis_showscale=False, yaxis_range=[0, 100], hovermode="x unified")
            st.plotly_chart(fig_upsell, use_container_width=True)
            st.caption("Red bars indicate days where a high percentage of customers bought less than the average amount. These days require sales training.")
        else:
            st.info("Date column not present to compute daily AOV stats.")

    # --- Category Details ---
    with tab_cat:
        st.header("📂 Category Deep-Dive")
        cats = sorted(df["Category"].unique()) if "Category" in df.columns else []
        total_business_rev = df["TotalAmount"].sum() if "TotalAmount" in df.columns else 0
        days_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        if cats:
            selected_cat_deep_dive = st.selectbox("Select Category to Analyze", cats, key="cat_deep_dive_select")
            cat_data = df[df["Category"] == selected_cat_deep_dive]
            st.subheader(f"🔹 {selected_cat_deep_dive}")
            if cat_data.empty:
                st.warning(f"No data for category {selected_cat_deep_dive}")
            else:
                cat_stats = cat_data.groupby("ItemName").agg({"TotalAmount": "sum", "Quantity": "sum"}).reset_index()
                cat_stats["Contribution %"] = (cat_stats["TotalAmount"] / total_business_rev) * 100 if total_business_rev > 0 else 0
                day_pivot = cat_data.groupby(["ItemName", "DayOfWeek"])["Quantity"].sum().unstack(fill_value=0)
                for day in days_order:
                    if day not in day_pivot.columns:
                        day_pivot[day] = 0
                day_pivot = day_pivot[days_order]
                cat_stats = pd.merge(cat_stats, day_pivot, on="ItemName", how="left").fillna(0)
                cat_stats = cat_stats.sort_values("TotalAmount", ascending=False)

                def mark_item_name(row):
                    name = row["ItemName"]
                    rev = row["TotalAmount"]
                    if name in pareto_list:
                        name = f"★ {name}"
                    if rev < 500:
                        name = f"{name} **"
                    elif 500 <= rev < 1500:
                        name = f"{name} *"
                    return name

                cat_stats["Item Name"] = cat_stats.apply(mark_item_name, axis=1)
                cols_to_show = ["Item Name", "TotalAmount", "Quantity", "Contribution %"] + days_order
                st.dataframe(cat_stats[cols_to_show], hide_index=True, use_container_width=True)
                st.caption("📝 **Legend:** `**` = Revenue < ₹500 (Critical) | `*` = Revenue < ₹1500 (Warning) | `★` = Pareto Top 80% Item")
                st.divider()

                # Hourly matrix
                st.subheader(f"Hourly Breakdown Matrix ({selected_cat_deep_dive})")
                st.markdown("Drill down: **Day** → **Date** → **Hourly Matrix**.")
                day_selected_mat = st.selectbox("1. Select Day of Week", days_order, key="mat_day_select")
                day_df_mat = df[df["DayOfWeek"] == day_selected_mat] if "DayOfWeek" in df.columns else pd.DataFrame()
                if not day_df_mat.empty:
                    unique_dates_mat = sorted(day_df_mat["Date"].dt.strftime("%Y-%m-%d").unique())
                    date_options_mat = [f"All {day_selected_mat}s Combined"] + unique_dates_mat
                    date_selected_mat = st.selectbox(f"2. Select Date ({day_selected_mat})", date_options_mat, key="mat_date_select")
                    if date_selected_mat == f"All {day_selected_mat}s Combined":
                        target_df_mat = day_df_mat
                    else:
                        target_df_mat = day_df_mat[day_df_mat["Date"].dt.strftime("%Y-%m-%d") == date_selected_mat]
                    target_df_mat = target_df_mat[target_df_mat["Category"] == selected_cat_deep_dive]
                    if not target_df_mat.empty:
                        if date_selected_mat == f"All {day_selected_mat}s Combined":
                            curr_idx = days_order.index(day_selected_mat)
                            prev_day = days_order[(curr_idx - 1) % 7]
                            next_day = days_order[(curr_idx + 1) % 7]
                            target_days = [prev_day, day_selected_mat, next_day]
                            df_3day_mat = df[df["DayOfWeek"].isin(target_days)]
                        else:
                            sel_dt_mat = pd.to_datetime(date_selected_mat)
                            target_dates_mat = [sel_dt_mat - timedelta(days=1), sel_dt_mat, sel_dt_mat + timedelta(days=1)]
                            df_3day_mat = df[df["Date"].isin(target_dates_mat)]
                        df_3day_mat = df_3day_mat[df_3day_mat["Category"] == selected_cat_deep_dive]
                        qty_3day_mat = df_3day_mat.groupby("ItemName")["Quantity"].sum().rename("3-Day Qty")
                        pivot_mat = target_df_mat.groupby(["ItemName", "Hour"])["Quantity"].sum().unstack(fill_value=0)
                        for h in range(9, 24):
                            if h not in pivot_mat.columns:
                                pivot_mat[h] = 0
                        pivot_mat = pivot_mat[sorted(pivot_mat.columns)]
                        pivot_mat.columns = [f"{int(h)}-{int(h)+1}" for h in pivot_mat.columns]
                        pivot_mat["Total Quantity"] = pivot_mat.sum(axis=1)
                        pivot_mat = pivot_mat.join(qty_3day_mat, how="left").fillna(0)
                        pivot_mat["3-Day Qty"] = pivot_mat["3-Day Qty"].astype(int)
                        pivot_mat = pivot_mat.sort_values("Total Quantity", ascending=False)
                        pivot_mat.index = pivot_mat.index.map(lambda x: f"★ {x}" if x in pareto_list else x)
                        st.dataframe(pivot_mat, use_container_width=True, height=600)
                    else:
                        st.warning(f"No data for {selected_cat_deep_dive} on this selection.")
                else:
                    st.warning(f"No transactions for {day_selected_mat}.")
        else:
            st.info("No categories found in uploaded data.")

    # --- Pareto tab ---
    with tab2:
        st.header("🏆 Pareto Analysis")
        pareto_df, ratio_msg, menu_perc = analyze_pareto_hierarchical(df)
        st.info(f"💡 **Insight:** {ratio_msg} (Only {menu_perc:.1f}% of your menu!)")
        if not pareto_df.empty:
            pareto_df["ItemName"] = pareto_df["ItemName"].apply(lambda x: f"★ {x}")
            st.dataframe(pareto_df, hide_index=True, use_container_width=True, height=600)
        else:
            st.info("No Pareto items to show.")

    # --- Time series tab ---
    with tab3:
        st.header("📅 Daily Trends")
        n_ts = st.slider("Number of items per category (Top N by Revenue)", 5, 30, 5)
        plot_time_series_fixed(df, pareto_list, n_ts)

    # --- Smart combos ---
    with tab4:
        st.header("🍔 Smart Combo Strategy")
        with st.expander("🛠️ Reorder Combo Layout"):
            combo_order = st.multiselect(
                "Section Order",
                ["Part 1: Full Combo Map", "Part 3: Strategic Recommendations"],
                default=["Part 1: Full Combo Map", "Part 3: Strategic Recommendations"],
            )

        def star_combo_string(item_name):
            return f"★ {item_name}" if item_name in pareto_list else item_name

        def process_combo_df_for_display(input_df):
            if input_df.empty:
                return input_df
            df_display = input_df.copy()
            if "Item A" in df_display.columns and "Item B" in df_display.columns:
                df_display["Item A Display"] = df_display["Item A"].apply(star_combo_string)
                df_display["Item B Display"] = df_display["Item B"].apply(star_combo_string)
                df_display["Specific Item Combo"] = df_display["Item A Display"] + " + " + df_display["Item B Display"]
            return df_display

        rules_display = process_combo_df_for_display(rules_df)
        proven_display = process_combo_df_for_display(proven_df)
        potential_display = process_combo_df_for_display(potential_df)

        def render_part1():
            st.subheader("1️⃣ Part 1: Full Category + Item Combo Map")
            if not rules_display.empty:
                display_cols = ["Category A", "Category B", "Specific Item Combo", "Times Sold Together", "Combo Value", "Peak Hour", "lift"]
                st.dataframe(rules_display[display_cols].sort_values("Times Sold Together", ascending=False), hide_index=True, use_container_width=True)
            else:
                st.warning("No significant combos found.")

        def render_part3():
            st.subheader("3️⃣ Part 3: Strategic Recommendations")
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("#### 🔥 Proven Winners")
                if not proven_display.empty:
                    st.dataframe(proven_display[["Specific Item Combo", "Times Sold Together", "Combo Value", "Peak Hour"]], hide_index=True, use_container_width=True)
                else:
                    st.info("No data.")
            with c2:
                st.markdown("#### 💎 Hidden Gems")
                if not potential_display.empty:
                    st.dataframe(potential_display[["Specific Item Combo", "lift", "Combo Value", "Peak Hour"]], hide_index=True, use_container_width=True)
                else:
                    st.info("No hidden gems found.")

        combo_map = {"Part 1: Full Combo Map": render_part1, "Part 3: Strategic Recommendations": render_part3}
        for block in combo_order:
            if block in combo_map:
                combo_map[block]()

    # --- Association Analysis (improved) ---
    with tab_assoc:
        st.header("🧬 Scientific Association Analysis (Enhanced)")
        st.markdown(
            """
        **Reliability Features:**
        - **Conviction**: Measures reliability (>1 is good)
        - **Leverage**: Positive indicates positive correlation
        - **Zhang's Metric**: -1..1 (higher positive => stronger)
        - **Reliability Score**: Composite score
        """
        )
        c1, c2, c3 = st.columns(3)
        with c1:
            analysis_level = st.radio("Analysis Level", ["ItemName", "Category"], horizontal=True)
        with c2:
            min_conviction_threshold = st.slider("Min Conviction (Reliability)", 1.0, 5.0, 1.2, step=0.1)
        with c3:
            min_leverage_threshold = st.slider("Min Leverage (Strength)", 0.0, 0.1, 0.01, step=0.01, format="%.3f")

        valid_df_global = df[df["Quantity"] > 0].copy()
        if valid_df_global.empty:
            st.warning("No valid transactions for association analysis.")
        else:
            basket_global = valid_df_global.groupby(["OrderID", analysis_level])["Quantity"].count().unstack().fillna(0)
            basket_sets_global = (basket_global > 0).astype(bool)
            supports = basket_sets_global.mean().sort_values(ascending=False)
            max_sup = supports.max() if not supports.empty else 1.0
            max_sup_percent = float(min(100.0, max_sup * 100 + 1.0))
            default_val = min(0.5, max_sup_percent / 2)
            min_support_percent = st.slider("Minimum Support (%)", 0.01, max_sup_percent, default_val, step=0.01, format="%.2f%%")
            min_support_val = min_support_percent / 100.0

            with st.spinner("Running Enhanced Association Analysis..."):
                frequent_itemsets = fpgrowth(basket_sets_global, min_support=min_support_val, use_colnames=True)
                if frequent_itemsets.empty:
                    assoc_rules = pd.DataFrame()
                else:
                    assoc_rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.1)
                    if assoc_rules.empty:
                        assoc_rules = pd.DataFrame()
                    else:
                        assoc_rules["No. of Orders"] = (assoc_rules["support"] * len(basket_sets_global)).round().astype(int)
                        assoc_rules["Support (%)"] = assoc_rules["support"] * 100
                        assoc_rules["zhang"] = assoc_rules.apply(
                            lambda row: (
                                (row["support"] - row["antecedent support"] * row["consequent support"])
                                / max(row["support"] * (1 - row["antecedent support"]), row["antecedent support"] * (row["consequent support"] - row["support"]), 1e-9)
                            ),
                            axis=1,
                        )
                        assoc_rules["Antecedent"] = assoc_rules["antecedents"].apply(lambda x: list(x)[0] if len(x) > 0 else "")
                        assoc_rules["Consequent"] = assoc_rules["consequents"].apply(lambda x: list(x)[0] if len(x) > 0 else "")
                        if analysis_level == "Category":
                            assoc_rules = assoc_rules[assoc_rules["Antecedent"] != assoc_rules["Consequent"]]

                        qty_basket_global = valid_df_global.groupby(["OrderID", analysis_level])["Quantity"].sum().unstack().fillna(0)

                        def calculate_split_quantity(row):
                            ant = row["Antecedent"]
                            con = row["Consequent"]
                            if ant not in qty_basket_global.columns or con not in qty_basket_global.columns:
                                return "0 (0 + 0)"
                            common_orders_mask = (qty_basket_global[ant] > 0) & (qty_basket_global[con] > 0)
                            sum_ant = int(qty_basket_global.loc[common_orders_mask, ant].sum())
                            sum_con = int(qty_basket_global.loc[common_orders_mask, con].sum())
                            total = sum_ant + sum_con
                            return f"{total} ({sum_ant} + {sum_con})"

                        assoc_rules["Total Item Qty"] = assoc_rules.apply(calculate_split_quantity, axis=1)
                        assoc_rules = ImprovedAssociationAnalyzer.calculate_all_metrics(assoc_rules)
                        # filter by user thresholds
                        assoc_rules_filtered = assoc_rules[
                            (assoc_rules["conviction"] >= min_conviction_threshold) & (assoc_rules["leverage"] >= min_leverage_threshold)
                        ].copy()
                        if not assoc_rules_filtered.empty:
                            assoc_rules = ImprovedAssociationAnalyzer.rank_by_reliability(assoc_rules_filtered)
                        else:
                            # fallback: show top ranked by reliability even if thresholds not met
                            assoc_rules = ImprovedAssociationAnalyzer.rank_by_reliability(assoc_rules).head(50)

                if assoc_rules.empty:
                    st.warning("No rules found with current parameters.")
                else:
                    # mark status (proven/potential)
                    def get_status(row):
                        if analysis_level != "ItemName":
                            return "Category Level"
                        pair = tuple(sorted([row["Antecedent"], row["Consequent"]]))
                        if pair in proven_list:
                            return "🔥 Proven Winner"
                        if pair in potential_list:
                            return "💎 Hidden Gem"
                        return "Normal"

                    assoc_rules["Status"] = assoc_rules.apply(get_status, axis=1)
                    if analysis_level == "ItemName":
                        assoc_rules["Antecedent"] = assoc_rules["Antecedent"].apply(lambda x: f"★ {x}" if x in pareto_list else x)
                        assoc_rules["Consequent"] = assoc_rules["Consequent"].apply(lambda x: f"★ {x}" if x in pareto_list else x)
                    display_cols = ["Status", "Antecedent", "Consequent", "Support (%)", "Total Item Qty", "confidence", "lift", "conviction", "leverage", "zhang", "reliability_score"]
                    # Some columns may be missing depending on association_rules output; filter
                    display_cols = [c for c in display_cols if c in assoc_rules.columns]
                    st.dataframe(assoc_rules[display_cols], use_container_width=True, height=600)
                    fig = px.scatter(assoc_rules, x="Support (%)", y="confidence", size="lift" if "lift" in assoc_rules.columns else None, color="reliability_score" if "reliability_score" in assoc_rules.columns else "zhang", hover_data=["Antecedent", "Consequent", "Total Item Qty"])
                    st.plotly_chart(fig, use_container_width=True)

    # --- Demand Forecast ---
    with tab5:
        st.header("🔮 Demand Prediction (Ensemble AI)")
        st.markdown("**Model:** Hybrid Ensemble (Prophet + XGBoost + Random Forest + Meta-Learner).")
        st.markdown("---")
        forecast_file = st.file_uploader("Upload Historical Master File (Recommended: 3+ Months Data)", type=["xlsx"], key="forecast_uploader")
        df_to_use = None
        if forecast_file:
            df_to_use = load_data(forecast_file)
            if df_to_use is not None:
                st.success("✅ Using Master History File for Training")
        else:
            df_to_use = df
            st.warning("⚠️ Using Sidebar File (Short-term data). Accuracy may be low without history.")
        if df_to_use is not None and not df_to_use.empty:
            all_items = sorted(df_to_use["ItemName"].unique())
            selected_item = st.selectbox("Select Item to Forecast", all_items)
            if st.button("Generate AI Forecast"):
                with st.spinner(f"Training AI Models for {selected_item}..."):
                    item_df = df_to_use[df_to_use["ItemName"] == selected_item].groupby("Date")["Quantity"].sum().reset_index()
                    item_df.columns = ["ds", "y"]
                    if len(item_df) < 14:
                        st.error(f"⚠️ Not enough history for {selected_item} (Need 14+ days). Model cannot train.")
                    else:
                        try:
                            forecaster = HybridDemandForecaster()
                            forecaster.fit(item_df)
                            forecast = forecaster.predict(periods=30)
                            st.subheader(f"Forecast for {selected_item}")
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(x=item_df["ds"], y=item_df["y"], mode="markers", name="Actual Sales", marker=dict(color="gray", opacity=0.5, size=8)))
                            fig.add_trace(go.Scatter(x=forecast["ds"], y=forecast["Predicted_Demand"], mode="lines+markers", name="Ensemble Prediction", line=dict(color="#00CC96", width=3)))
                            fig.add_trace(go.Scatter(x=forecast["ds"], y=forecast["Prophet_View"], mode="lines", name="Prophet View", line=dict(dash="dot", color="blue", width=1), visible="legendonly"))
                            fig.add_trace(go.Scatter(x=forecast["ds"], y=forecast["XGB_View"], mode="lines", name="XGBoost View", line=dict(dash="dot", color="red", width=1), visible="legendonly"))
                            fig.update_layout(title="30-Day Demand Forecast", xaxis_title="Date", yaxis_title="Predicted Quantity", template="plotly_white")
                            st.plotly_chart(fig, use_container_width=True)
                            st.markdown("#### Detailed Forecast Data")
                            st.dataframe(forecast[["ds", "Predicted_Demand", "Prophet_View", "XGB_View"]], use_container_width=True)
                        except Exception as e:
                            st.error(f"Modeling Error: {e}")
        else:
            st.info("Please upload data to begin forecasting.")

    # --- AI Chat ---
    with tab6:
        st.subheader("🤖 Manager Chat")
        if "messages" not in st.session_state:
            st.session_state["messages"] = []
        for msg in st.session_state["messages"]:
            st.chat_message(msg["role"]).write(msg["content"])
        prompt = st.chat_input("Ask about combos or forecasts...")
        if prompt:
            st.chat_message("user").write(prompt)
            st.session_state["messages"].append({"role": "user", "content": prompt})
            if not LLM_AVAILABLE:
                st.error("LLM/chat integration not available in this environment.")
            else:
                try:
                    llm = ChatOpenAI(model="gpt-4o-mini", api_key=st.secrets.get("OPENAI_API_KEY"))
                    response = llm.invoke([SystemMessage(content="Restaurant Analyst"), HumanMessage(content=prompt)])
                    assistant_text = response.content if response and hasattr(response, "content") else str(response)
                    st.chat_message("assistant").write(assistant_text)
                    st.session_state["messages"].append({"role": "assistant", "content": assistant_text})
                except Exception:
                    st.error("LLM call failed. Check configuration or API key.")
else:
    st.info("👋 Upload data to begin.")
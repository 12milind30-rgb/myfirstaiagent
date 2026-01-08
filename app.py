import base64
import io
import datetime
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State, dash_table, callback_context
import dash_bootstrap_components as dbc
import warnings

warnings.filterwarnings('ignore')

# --- APP CONFIGURATION ---
# We use a 'Darkly' theme for that professional dark mode look
app = Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
server = app.server

# --- DATA PROCESSING HELPERS ---
def parse_contents(contents, filename):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        if 'xlsx' in filename:
            df = pd.read_excel(io.BytesIO(decoded))
        else:
            return None
            
        # Standardize Columns
        col_map = {
            'Invoice No.': 'OrderID', 'Item Name': 'ItemName', 'Qty.': 'Quantity',
            'Final Total': 'TotalAmount', 'Price': 'UnitPrice', 'Category': 'Category',
            'Timestamp': 'Time', 'Date': 'Date'
        }
        df = df.rename(columns=col_map)
        
        # Cleanup
        for c in ['Quantity', 'TotalAmount', 'UnitPrice']:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            df['DayOfWeek'] = df['Date'].dt.day_name()
            df['DateStr'] = df['Date'].dt.strftime('%Y-%m-%d') # Needed for JSON serialization
        
        if 'Time' in df.columns:
            try:
                # Extract hour safely
                df['Hour'] = pd.to_datetime(df['Time'].astype(str), format='%H:%M:%S', errors='coerce').dt.hour
                if df['Hour'].isnull().all():
                     df['Hour'] = pd.to_datetime(df['Time'], errors='coerce').dt.hour
                df['Hour'] = df['Hour'].fillna(0).astype(int)
            except:
                df['Hour'] = 0
                
        return df.to_dict('records') # Dash stores data as JSON
    except Exception as e:
        print(e)
        return None

# --- LAYOUT (The HTML Structure) ---
app.layout = dbc.Container([
    # Store data in browser session (Invisible)
    dcc.Store(id='stored-data'),
    
    # Header
    dbc.Row([
        dbc.Col(html.H1("📊 Mithas Intelligence Dashboard", className="text-center mb-4"), width=12)
    ], className="mt-4"),
    
    # Upload Section
    dbc.Row([
        dbc.Col([
            dcc.Upload(
                id='upload-data',
                children=html.Div([
                    'Drag and Drop or ',
                    html.A('Select Monthly Excel File')
                ]),
                style={
                    'width': '100%', 'height': '60px', 'lineHeight': '60px',
                    'borderWidth': '1px', 'borderStyle': 'dashed',
                    'borderRadius': '5px', 'textAlign': 'center', 'margin': '10px'
                },
                multiple=False
            ),
            html.Div(id='upload-status', className="text-center text-success")
        ], width={"size": 8, "offset": 2})
    ]),
    
    html.Hr(),
    
    # Main Tabs
    dbc.Tabs([
        dbc.Tab(label="Overview", tab_id="tab-overview"),
        dbc.Tab(label="Category Details", tab_id="tab-category"),
        dbc.Tab(label="Time Series", tab_id="tab-timeseries"),
        # We will add the other tabs (Pareto, Combos, Chat) once this is working
    ], id="tabs", active_tab="tab-overview"),
    
    # Content Area
    html.Div(id="tab-content", className="p-4")
    
], fluid=True)

# --- CALLBACKS (The Logic) ---

# 1. Handle Upload & Data Storage
@app.callback(
    [Output('stored-data', 'data'), Output('upload-status', 'children')],
    Input('upload-data', 'contents'),
    State('upload-data', 'filename')
)
def update_output(contents, filename):
    if contents is None:
        return None, ""
    data = parse_contents(contents, filename)
    if data:
        return data, f"✅ Successfully loaded: {filename}"
    return None, "❌ Error loading file"

# 2. Render Tabs
@app.callback(
    Output("tab-content", "children"),
    [Input("tabs", "active_tab"), Input("stored-data", "data")]
)
def render_tab_content(active_tab, data):
    if not data:
        return html.Div("Please upload a file to view analytics.", className="text-center text-warning")
    
    # Convert JSON back to DataFrame
    df = pd.DataFrame(data)
    
    if active_tab == "tab-overview":
        return render_overview(df)
    elif active_tab == "tab-category":
        return render_category(df)
    elif active_tab == "tab-timeseries":
        return render_timeseries(df)
    
    return html.P("This tab is under construction.")

# --- RENDER FUNCTIONS (UI Logic) ---

def render_overview(df):
    # Metrics
    total_rev = df['TotalAmount'].sum()
    total_orders = df['OrderID'].nunique()
    avg_rev_day = total_rev / df['DateStr'].nunique() if df['DateStr'].nunique() else 0
    
    # Peak Hours Graph
    hourly = df.groupby('Hour')['TotalAmount'].sum().reset_index()
    fig_hourly = px.bar(hourly, x='Hour', y='TotalAmount', title="Peak Hours", template="plotly_dark")
    fig_hourly.update_xaxes(tickmode='linear', dtick=1)
    
    # Peak Days Graph
    daily = df.groupby('DayOfWeek')['TotalAmount'].sum().reset_index()
    fig_daily = px.bar(daily, x='DayOfWeek', y='TotalAmount', title="Peak Days", template="plotly_dark")
    
    # Star Items
    item_stats = df.groupby('ItemName').agg({'TotalAmount': 'sum'}).reset_index()
    item_stats['Contribution %'] = (item_stats['TotalAmount'] / total_rev) * 100
    star_items = item_stats.sort_values('TotalAmount', ascending=False).head(20)
    
    return html.Div([
        # Metrics Row
        dbc.Row([
            dbc.Col(dbc.Card(dbc.CardBody([html.H4("💰 Total Revenue"), html.H2(f"₹{total_rev:,.0f}")])), width=4),
            dbc.Col(dbc.Card(dbc.CardBody([html.H4("🧾 Orders"), html.H2(f"{total_orders}")])), width=4),
            dbc.Col(dbc.Card(dbc.CardBody([html.H4("📅 Avg Daily Rev"), html.H2(f"₹{avg_rev_day:,.0f}")])), width=4),
        ], className="mb-4"),
        
        # Graphs Row
        dbc.Row([
            dbc.Col(dcc.Graph(figure=fig_hourly), width=6),
            dbc.Col(dcc.Graph(figure=fig_daily), width=6),
        ]),
        
        html.Hr(),
        html.H3("⭐ Top 20 Star Items"),
        
        # Dash DataTable for Star Items
        dash_table.DataTable(
            data=star_items.to_dict('records'),
            columns=[
                {"name": "Item Name", "id": "ItemName"},
                {"name": "Revenue (₹)", "id": "TotalAmount", "type": "numeric", "format": {"specifier": ",.0f"}},
                {"name": "Contrib %", "id": "Contribution %", "type": "numeric", "format": {"specifier": ".2f"}}
            ],
            style_header={'backgroundColor': 'rgb(30, 30, 30)', 'color': 'white'},
            style_data={'backgroundColor': 'rgb(50, 50, 50)', 'color': 'white'},
            page_size=10
        )
    ])

def render_category(df):
    cats = df['Category'].unique()
    total_rev = df['TotalAmount'].sum()
    
    layout = [html.H2("Category Deep Dive")]
    
    for cat in cats:
        cat_data = df[df['Category'] == cat]
        stats = cat_data.groupby('ItemName').agg({'TotalAmount': 'sum', 'Quantity': 'sum'}).reset_index()
        stats['Contrib'] = (stats['TotalAmount'] / total_rev) * 100
        stats = stats.sort_values('TotalAmount', ascending=False)
        
        layout.append(html.H4(f"🔹 {cat}", className="mt-4"))
        layout.append(dash_table.DataTable(
            data=stats.to_dict('records'),
            columns=[
                {"name": "Item", "id": "ItemName"},
                {"name": "Revenue", "id": "TotalAmount", "format": {"specifier": ",.0f"}},
                {"name": "Units", "id": "Quantity"},
                {"name": "Global Share %", "id": "Contrib", "format": {"specifier": ".2f"}}
            ],
            style_header={'backgroundColor': '#2c3e50', 'color': 'white'},
            style_data={'backgroundColor': '#34495e', 'color': 'white'},
            style_cell={'textAlign': 'left'},
        ))
        
    return html.Div(layout)

def render_timeseries(df):
    fig = px.line(df.groupby('DateStr')['TotalAmount'].sum().reset_index(), 
                  x='DateStr', y='TotalAmount', title="Daily Revenue Trend", template="plotly_dark")
    return dcc.Graph(figure=fig)

if __name__ == '__main__':
    app.run(debug=True)
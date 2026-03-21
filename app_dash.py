import dash
from dash import dcc, html, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ── 1. LOAD & INITIALIZE DATA ────────────────────────────────────────────────
def load_data():
    df = pd.read_csv("/Users/yeswanth/Desktop/VA/Dataset/Visual_Analytics/cleaned_air_quality_merged.csv")
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").dropna().reset_index(drop=True)
    if len(df) > 5000:
        df = df.tail(5000).reset_index(drop=True)
    return df

df_full = load_data()
pollutants = [col for col in df_full.select_dtypes(include=[np.number]).columns if col != "datetime"]

# ── 2. DASH APP SETUP ────────────────────────────────────────────────────────
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG, "https://fonts.googleapis.com/css2?family=Inter:wght@400;700&display=swap"])
app.title = "Air Quality Analytics | Dash"

# ── 3. ASYNC MODEL WRAPPER ───────────────────────────────────────────────────
# In a real app, we'd cache this more aggressively. Here we rely on persistent variables for the demo.
def get_model_and_preds(target, selected_features, n_trees, max_depth):
    X = df_full[selected_features]
    y = df_full[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)
    
    model = RandomForestRegressor(n_estimators=n_trees, max_depth=max_depth, max_features="sqrt", n_jobs=-1, random_state=42)
    model.fit(X_train, y_train)
    
    X_test_reset = X_test.reset_index(drop=True)
    y_test_reset = y_test.reset_index(drop=True)
    test_dates = df_full.loc[X_test.index, "datetime"].reset_index(drop=True)
    
    orig_pred = model.predict(X_test_reset)
    # Get per-tree predictions: (n_trees, n_test_samples)
    tree_preds = np.array([t.predict(X_test_reset) for t in model.estimators_])
    
    return model, tree_preds, X_test_reset, y_test_reset, test_dates, orig_pred

# ── 4. LAYOUT DEFINITION ─────────────────────────────────────────────────────
app.layout = dbc.Container([
    # Initial Calculation Store
    dcc.Store(id='model-data-store'),
    dcc.Store(id='disabled-trees-store', data=[]),
    dcc.Store(id='selection-idx-store', data=0),

    dbc.Row([
        dbc.Col(html.H1("🌿 Air Quality Analytics Dashboard", className="text-center py-4", style={'fontFamily': 'Inter'}), width=12)
    ]),

    dbc.Row([
        # Sidebar Panel
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Model Configuration"),
                dbc.CardBody([
                    html.Label("Target Pollutant"),
                    dcc.Dropdown(id='target-selector', options=[{'label': p, 'value': p} for p in pollutants], value="n02_palmes"),
                    html.Br(),
                    html.Label("Predictors"),
                    dcc.Dropdown(id='feature-selector', multi=True),
                    html.Br(),
                    html.Label("Number of Trees"),
                    dcc.Slider(5, 30, 1, value=15, id='slider-n-trees', marks={i: str(i) for i in range(5, 31, 5)}),
                    html.Br(),
                    html.Label("Tree Max Depth"),
                    dcc.Slider(1, 30, 1, value=12, id='slider-max-depth', marks={i: str(i) for i in range(0, 31, 5)}),
                    html.Hr(),
                    dbc.Button("Apply & Retrain", id='btn-retrain', color="primary", className="w-100")
                ])
            ], color="dark", outline=True)
        ], width=3),

        # Main Performance and Forecast
        dbc.Col([
            dbc.Row([
                dbc.Col(dbc.Card([dbc.CardBody([html.H6("R² Score"), html.H3(id="metric-r2", className="text-info")])]), width=4),
                dbc.Col(dbc.Card([dbc.CardBody([html.H6("MAE"), html.H3(id="metric-mae", className="text-warning")])]), width=4),
                dbc.Col(dbc.Card([dbc.CardBody([html.H6("MSE"), html.H3(id="metric-mse", className="text-danger")])]), width=4),
            ], className="mb-3"),
            
            dbc.Card([
                dbc.CardBody([
                    dcc.Graph(id='forecast-graph', config={'displayModeBar': False}),
                    html.P("Slide to inspect details across timeline:", className="mt-3 text-muted small"),
                    dcc.Slider(id='timeline-slider', step=1)
                ])
            ], color="secondary", outline=True, className="p-2")
        ], width=9)
    ], className="mb-4"),

    # Live Insight Banner
    dbc.Row([
        dbc.Col(html.Div(id='insight-banner'), width=12)
    ], className="mb-4"),

    # Bottom Interaction Section
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("🌳 Tree Votes & Correlation"),
                dbc.CardBody([
                    dcc.Graph(id='heatmap-graph', config={'displayModeBar': False})
                ])
            ])
        ], width=6),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("🌲 Individual Tree Decisions"),
                dbc.CardBody([
                    dcc.Graph(id='tree-bar-graph', config={'displayModeBar': False})
                ])
            ])
        ], width=6)
    ]),

    # Quick Actions
    dbc.Row([
        dbc.Col([
            html.Div([
                dbc.Button("✅ Enable All", id='btn-enable-all', color="success", size="sm", className="me-2"),
                dbc.Button("⛔ Disable Worst 5", id='btn-disable-worst', color="danger", size="sm", className="me-2"),
                dbc.Button("🔁 Reset Selection", id='btn-reset-pos', color="info", size="sm")
            ], className="d-flex justify-content-center py-4")
        ], width=12)
    ])
], fluid=True, style={'backgroundColor': '#0b0f19', 'minHeight': '100vh', 'color': '#e2e8f0'})

# ── 5. CALLBACKS ─────────────────────────────────────────────────────────────

# Sync feature selector options based on target
@app.callback(
    Output('feature-selector', 'options'),
    Output('feature-selector', 'value'),
    Input('target-selector', 'value')
)
def update_feature_list(target):
    feats = [p for p in pollutants if p != target]
    return [{'label': f, 'value': f} for f in feats], feats[:4]

# Master Data Store Controller (Retrain)
@app.callback(
    Output('model-data-store', 'data'),
    Output('timeline-slider', 'max'),
    Input('btn-retrain', 'n_clicks'),
    State('target-selector', 'value'),
    State('feature-selector', 'value'),
    State('slider-n-trees', 'value'),
    State('slider-max-depth', 'value')
)
def handle_retrain(n, target, features, n_trees, depth):
    if not features: return dash.no_update, dash.no_update
    model, t_preds, X_test, y_test, dates, o_pred = get_model_and_preds(target, features, n_trees, depth)
    
    # Store essential data for callbacks (as dict/json)
    stored_data = {
        'tree_preds': t_preds.tolist(),
        'y_test': y_test.tolist(),
        'orig_pred': o_pred.tolist(),
        'dates': [d.strftime("%Y-%m-%d %H:%M") for d in dates],
        'n_total_trees': n_trees,
        'features': features,
        'target': target
    }
    return stored_data, len(y_test) - 1

# Handle Selection Sync (Click -> Slider -> Selection Store)
@app.callback(
    Output('selection-idx-store', 'data'),
    Output('timeline-slider', 'value'),
    Input('forecast-graph', 'clickData'),
    Input('timeline-slider', 'value'),
    Input('btn-reset-pos', 'n_clicks'),
    State('selection-idx-store', 'data'),
    prevent_initial_call=True
)
def sync_selection(click, slider_val, reset_n, current_store):
    ctx = callback_context
    if not ctx.triggered: return 0, 0
    trigger_id = ctx.triggered[0]['prop_id']
    
    if 'btn-reset-pos' in trigger_id: return 0, 0
    if 'forecast-graph' in trigger_id:
        idx = click['points'][0]['pointIndex']
        return idx, idx
    return slider_val, slider_val

# Handle Tree Toggling Store
@app.callback(
    Output('disabled-trees-store', 'data'),
    Input('heatmap-graph', 'clickData'),
    Input('tree-bar-graph', 'clickData'),
    Input('btn-enable-all', 'n_clicks'),
    Input('btn-disable-worst', 'n_clicks'),
    State('disabled-trees-store', 'data'),
    State('model-data-store', 'data'),
    State('selection-idx-store', 'data'),
    prevent_initial_call=True
)
def update_disabled_trees(heat_click, bar_click, enable_n, disable_worst_n, current_list, model_data, sel_idx):
    if not model_data: return []
    ctx = callback_context
    trigger = ctx.triggered[0]['prop_id']
    
    new_list = set(current_list)
    
    if 'btn-enable-all' in trigger:
        return []
    
    if 'btn-disable-worst' in trigger:
        tp = np.array(model_data['tree_preds'])
        # preds_at_spike has shape (n_trees,) for the current point
        preds_at_spike = tp[:, sel_idx]
        actual = model_data['y_test'][sel_idx]
        # Find 5 trees with highest absolute error vs actual
        errs = np.abs(preds_at_spike - actual)
        worst_indices = np.argsort(errs)[-5:]
        return list(set(worst_indices))

    if 'heatmap-graph' in trigger:
        # For Heatmap, x is the label we provided: T0, T1, or 🔴
        # We can also use pointNumber[1] which is the x-coordinate (ti)
        ti = heat_click['points'][0]['pointNumber'][1] 
        if ti in new_list: new_list.discard(ti)
        else: new_list.add(ti)
        
    if 'tree-bar-graph' in trigger:
        # For Bar, pointNumber is the index in the data array
        ti = bar_click['points'][0]['pointNumber']
        if ti in new_list: new_list.discard(ti)
        else: new_list.add(ti)
        
    return list(new_list)

# Main Multi-Output Graphics Update
@app.callback(
    Output('forecast-graph', 'figure'),
    Output('heatmap-graph', 'figure'),
    Output('tree-bar-graph', 'figure'),
    Output('insight-banner', 'children'),
    Output('metric-r2', 'children'),
    Output('metric-mae', 'children'),
    Output('metric-mse', 'children'),
    Input('model-data-store', 'data'),
    Input('disabled-trees-store', 'data'),
    Input('selection-idx-store', 'data')
)
def update_all_plots(data, disabled, sel_idx):
    if not data: return go.Figure(), go.Figure(), go.Figure(), "", "--", "--", "--"
    
    t_preds = np.array(data['tree_preds'])
    y_test = np.array(data['y_test'])
    orig_pred = np.array(data['orig_pred'])
    dates = data['dates']
    n_total = data['n_total_trees']
    
    # Calculate Modified Pred
    mask = np.ones(n_total, dtype=bool)
    for ti in disabled: 
        if ti < n_total: mask[ti] = False
    
    active_preds = t_preds[mask]
    if active_preds.shape[0] > 0:
        mod_pred = active_preds.mean(axis=0)
    else:
        mod_pred = orig_pred
        
    n_disabled = len(disabled)
    
    # ── METRICS ──
    r2 = r2_score(y_test, mod_pred)
    mae = mean_absolute_error(y_test, mod_pred)
    mse = mean_squared_error(y_test, mod_pred)

    # ── FORECAST PLOT ──
    fig_ts = go.Figure()
    fig_ts.add_trace(go.Scatter(x=dates, y=y_test, name="Actual", line=dict(color="#38bdf8", width=2)))
    fig_ts.add_trace(go.Scatter(x=dates, y=orig_pred, name="Original Pred", line=dict(color="#ea580c", width=2)))
    if n_disabled > 0:
        fig_ts.add_trace(go.Scatter(x=dates, y=mod_pred, name="Modified Pred", line=dict(color="#8b5cf6", width=2.5)))
    
    # Highlight selection
    fig_ts.add_trace(go.Scatter(x=[dates[sel_idx]], y=[orig_pred[sel_idx]], mode="markers", 
                                marker=dict(color="#fff", size=14, line=dict(color="#000", width=2)), showlegend=False))
    
    fig_ts.update_layout(template="plotly_dark", height=450, margin=dict(l=10, r=10, t=30, b=10),
                         paper_bgcolor="#0b0f19", plot_bgcolor="#0b0f19",
                         legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))

    # ── HEATMAP PLOT ──
    max_h = min(30, n_total)
    feats = data['features']
    # Windowed correlation (±50 pts)
    w_start = max(0, sel_idx - 50); w_end = min(len(y_test), sel_idx + 51)
    corr_matrix = np.zeros((len(feats), max_h))
    for fi, f_name in enumerate(feats):
        # We don't have X_test in data store to keep it small, but for Dash demo we assume features exist in df_full
        # This is a simplification: usually we'd store the window in the Store or re-calculate.
        # Let's approximate or just use the per-tree variability vs feature window
        fv = df_full.loc[w_start:w_end-1, f_name].values
        for ti in range(max_h):
            tv = t_preds[ti, w_start:w_end]
            if np.std(fv) > 1e-6 and np.std(tv) > 1e-6:
                corr_matrix[fi, ti] = np.corrcoef(fv, tv)[0, 1]
    
    labels = [f"T{i}" if i not in disabled else "🔴" for i in range(max_h)]
    
    # Rich hover text logic
    hover_text = []
    tree_votes_at_spike = t_preds[:, sel_idx]
    for fi, f_name in enumerate(feats):
        row_txt = []
        for ti in range(max_h):
            c_val = corr_matrix[fi, ti]
            v_val = tree_votes_at_spike[ti]
            status = "⛔ DISABLED" if ti in disabled else "✅ Active"
            strength = "strong" if abs(c_val) > 0.5 else "moderate" if abs(c_val) > 0.2 else "weak"
            row_txt.append(
                f"<b>Tree T{ti} × {f_name}</b><br>Correlation: {c_val:.3f} ({strength})<br>"
                f"Status: {status}<br>Current Vote: {v_val:.3f}"
            )
        hover_text.append(row_txt)

    fig_heat = go.Figure()
    # Base heatmap
    fig_heat.add_trace(go.Heatmap(
        z=corr_matrix, x=labels, y=feats, 
        colorscale="RdYlGn", zmid=0, zmin=-1, zmax=1,
        text=hover_text, hoverinfo="text"
    ))
    
    # Overlay for disabled columns
    if disabled:
        overlay_z = np.zeros((len(feats), max_h))
        for ti in disabled:
            if ti < max_h: overlay_z[:, ti] = 1.0
        fig_heat.add_trace(go.Heatmap(
            z=overlay_z, x=labels, y=feats,
            colorscale=[[0, "rgba(0,0,0,0)"], [1, "rgba(244,63,94,0.22)"]],
            showscale=False, hoverinfo="skip"
        ))

    fig_heat.update_layout(template="plotly_dark", height=380, margin=dict(l=10, r=10, t=10, b=10),
                           xaxis=dict(title="Tree Index"), yaxis=dict(title="Features"))

    # ── TREE BAR PLOT ──
    colors = ["#f43f5e" if i in disabled else "#34d399" for i in range(max_h)]
    fig_tree = go.Figure(go.Bar(x=[f"T{i}" for i in range(max_h)], y=t_preds[:max_h, sel_idx], marker_color=colors))
    fig_tree.add_hline(y=y_test[sel_idx], line_dash="dash", line_color="#38bdf8", annotation_text="Actual")
    fig_tree.update_layout(template="plotly_dark", height=380, margin=dict(l=10, r=10, t=10, b=10))

    # ── INSIGHT BANNER ──
    banner = dbc.Alert([
        html.B("🕒 Selection: "), dates[sel_idx],
        html.Span(f" | Actual: {y_test[sel_idx]:.3f}", className="ms-3"),
        html.Span(f" | Mod Pred: {mod_pred[sel_idx]:.3f}", className="ms-3", style={'color': '#a78bfa'}),
        html.Span(f" | Active Trees: {n_total - n_disabled}/{n_total}", className="ms-3")
    ], color="dark", style={'borderLeft': '4px solid #facc15'})

    return fig_ts, fig_heat, fig_tree, banner, f"{r2:.3f}", f"{mae:.3f}", f"{mse:.3f}"

if __name__ == '__main__':
    app.run(debug=True, port=8050)

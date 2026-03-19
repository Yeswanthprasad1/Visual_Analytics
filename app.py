import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import shap
from sklearn.tree import plot_tree

st.set_page_config(layout="wide", page_title="Air Quality Visual Analytics Dashboard")

st.title("Air Quality Visual Analytics Dashboard")

# -----------------------------
# 1. LOAD DATA
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("/Users/yeswanth/Desktop/VA/Dataset/Visual_Analytics/cleaned_air_quality_merged.csv")
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").dropna().reset_index(drop=True)
    # Use last 5000 rows for interactive performance
    if len(df) > 5000:
        df = df.tail(5000).reset_index(drop=True)
    return df

df = load_data()
# Only consider numeric pollutant columns (exclude datetime)
pollutants = [col for col in df.select_dtypes(include=[np.number]).columns if col != "datetime"]

# ---------- APP NAVIGATION ----------
view = st.sidebar.radio("App View", ["Forecasting & Tuning", "Correlation Explorer"], index=0)
if view == "Correlation Explorer":
    try:
        from correlation_view import render_relation
        render_relation()
    except Exception as e:
        st.error(f"Unable to load correlation explorer: {e}")
    st.stop()

# -----------------------------
# 5. MODEL IMPROVEMENT PANEL (SIDEBAR)
# -----------------------------
st.sidebar.header("Model Improvement Panel")

# Prefer 'n02_palmes' as default target if available
default_target = "n02_palmes"
target = st.sidebar.selectbox("Target Pollutant to Forecast", pollutants, index=pollutants.index(default_target) if default_target in pollutants else 0)

available_features = [p for p in pollutants if p != target]
selected_features = st.sidebar.multiselect("Input Predictors (Features)", available_features, default=available_features[:4])

# Default if no features selected
if not selected_features:
    st.warning("Please select at least one predictor feature in the sidebar.")
    st.stop()

# Prepare features and time-series split early so tuning can run before sliders are created
features_df = df[selected_features]
y = df[target]

# split while preserving time order
X_train, X_test, y_train, y_test = train_test_split(features_df, y, test_size=0.2, random_state=42, shuffle=False)
test_dates = df.loc[X_test.index, "datetime"]

# Tuning objective selection (use R² or MSE as requested)
# (keep it in sidebar so user can pick before selecting auto-tune)
tuning_obj = st.sidebar.radio("Tuning Objective", options=['R2 (maximize)', 'MSE (minimize)'], index=0)

# Mode selector replaces the button: running the grid when user picks 'Finding the Best Prediction'
mode = st.sidebar.selectbox("Mode", options=['Manual', 'Finding the Best Prediction'], index=0)

# Persist slider defaults in session_state so we can set them before widget creation
if 'n_trees' not in st.session_state:
    st.session_state['n_trees'] = 50
if 'max_depth' not in st.session_state:
    st.session_state['max_depth'] = 6

# If user selected auto-tune mode, run the grid search now (before sliders are created)
if mode == 'Finding the Best Prediction':
    st.sidebar.info("Running grid search — this may take a little while...")
    n_list = [10, 50, 100, 200]
    depth_list = [3, 6, 10, 15]

    results = []
    best_model = None
    best_pred = None
    best_params = (st.session_state['n_trees'], st.session_state['max_depth'])

    maximize_r2 = tuning_obj.startswith('R2')
    best_score = -np.inf if maximize_r2 else np.inf

    with st.spinner("Searching for best hyperparameters..."):
        for nt in n_list:
            for md in depth_list:
                clf = RandomForestRegressor(n_estimators=nt, max_depth=md, random_state=42, n_jobs=-1)
                clf.fit(X_train, y_train)
                yhat = clf.predict(X_test)
                mae_tmp = mean_absolute_error(y_test, yhat)
                r2_tmp = r2_score(y_test, yhat)
                mse_tmp = mean_squared_error(y_test, yhat)
                results.append({"n_estimators": nt, "max_depth": md, "MAE": mae_tmp, "MSE": mse_tmp, "R2": r2_tmp})

                if maximize_r2:
                    if r2_tmp > best_score:
                        best_score = r2_tmp
                        best_model = clf
                        best_pred = yhat
                        best_params = (nt, md)
                else:
                    if mse_tmp < best_score:
                        best_score = mse_tmp
                        best_model = clf
                        best_pred = yhat
                        best_params = (nt, md)

    res_df = pd.DataFrame(results)
    if maximize_r2:
        res_df = res_df.sort_values('R2', ascending=False).reset_index(drop=True)
    else:
        res_df = res_df.sort_values('MSE', ascending=True).reset_index(drop=True)

    # Apply best params into session_state BEFORE sliders are created
    best_n, best_md = best_params
    st.session_state['n_trees'] = int(best_n)
    st.session_state['max_depth'] = int(best_md)

    # save summary/results for display later
    mse = mean_squared_error(y_test, best_pred)
    mae = mean_absolute_error(y_test, best_pred)
    r2 = r2_score(y_test, best_pred)
    st.session_state['best_summary'] = {'mse': float(mse), 'mae': float(mae), 'r2': float(r2), 'maximize_r2': bool(maximize_r2)}
    st.session_state['best_res_df'] = res_df.head(10).to_dict()

    # set model/pred to the chosen best so main flow can render plots immediately
    model = best_model
    pred = best_pred

# Now create the sliders (their defaults come from session_state which may have been updated above)
# Set minimum number of trees to 11 as requested
n_trees = st.sidebar.slider("Number of Trees", min_value=11, max_value=200, value=st.session_state['n_trees'], step=10, key='n_trees')
max_depth = st.sidebar.slider("Tree Maximum Depth", min_value=1, max_value=20, value=st.session_state['max_depth'], step=1, key='max_depth')

# If manual mode, train with chosen slider values; if auto-tune mode we already set model/pred above
if mode == 'Manual':
    model = RandomForestRegressor(n_estimators=n_trees, max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    mse = mean_squared_error(y_test, pred)
    mae = mean_absolute_error(y_test, pred)
    r2 = r2_score(y_test, pred)

st.subheader("Model Performance Metrics")
m1, m2, m3 = st.columns(3)
m1.metric("R² Score", f"{r2:.3f}")
m2.metric("Mean Absolute Error", f"{mae:.3f}")
m3.metric("Mean Squared Error", f"{mse:.3f}")
st.markdown("---")

# If an auto-tune run just completed, display its summary and the top grid results saved in st.session_state
if mode == 'Finding the Best Prediction' and 'best_summary' in st.session_state and 'best_res_df' in st.session_state:
    best_summary = st.session_state['best_summary']
    best_df = pd.DataFrame(st.session_state['best_res_df'])
    st.info("Auto-tune was applied. Below are the summary metrics from the best found model:")
    b1, b2, b3 = st.columns(3)
    b1.metric("Chosen n_estimators", str(st.session_state.get('n_trees')))
    b2.metric("Chosen max_depth", str(st.session_state.get('max_depth')))
    if best_summary.get('maximize_r2'):
        b3.metric("Best R²", f"{best_summary['r2']:.3f}")
    else:
        b3.metric("Best MSE", f"{best_summary['mse']:.3f}")
    st.markdown("Top grid search results (by chosen objective):")
    # reconstruct dataframe (columns may be nested due to to_dict structure) -> transpose if needed
    try:
        display_df = pd.DataFrame(best_df)
    except Exception:
        display_df = pd.DataFrame.from_dict(st.session_state['best_res_df'])
    st.dataframe(display_df)

# Make the main time-series forecast visualization full width
st.header("Time Series Forecast Visualization")

# Build dataframe for plotting
ts_df = pd.DataFrame({"datetime": test_dates, "Actual": y_test, "Predicted": pred})

fig_ts = go.Figure()
fig_ts.add_trace(go.Scatter(x=ts_df["datetime"], y=ts_df["Actual"], mode='lines', name='Actual', line=dict(color='#1f77b4', width=2)))
fig_ts.add_trace(go.Scatter(x=ts_df["datetime"], y=ts_df["Predicted"], mode='lines', name='Predicted', line=dict(color='#ff7f0e', width=2, dash='dot')))
fig_ts.update_layout(
    title="Actual vs Predicted Pollutant Levels",
    xaxis_title="Time",
    yaxis_title=target,
    height=450,
    template="plotly_white",
    margin=dict(l=40, r=40, t=40, b=40),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)

# Render the time series chart full width
st.plotly_chart(fig_ts, use_container_width=True)

# Below the full-width chart, show error analysis and a quick top-errors table in two columns
c1, c2 = st.columns(2)

with c1:
    st.header("Prediction Error Analysis")
    error_df = pd.DataFrame({"Actual": y_test, "Predicted": pred, "Error": y_test - pred, "AbsError": np.abs(y_test - pred)})
    fig_err = px.histogram(error_df, x="Error", nbins=40, title="Error Distribution (Residuals)", color_discrete_sequence=['#ef553b'], template="plotly_white")
    fig_err.update_layout(height=400, margin=dict(l=40, r=40, t=40, b=40))
    fig_err.update_traces(marker_line_width=1, marker_line_color="black")
    st.plotly_chart(fig_err, use_container_width=True)

with c2:
    st.header("Top 5 Absolute Errors")
    # Show the rows with largest absolute errors to help diagnose model failures
    top_errors = error_df.sort_values("AbsError", ascending=False).head(5)
    # Keep the table compact and readable
    st.table(top_errors[["Actual", "Predicted", "AbsError"]].reset_index(drop=True))

# -----------------------------
# 3. FEATURE IMPORTANCE VISUALIZATION (SHAP)
# -----------------------------
st.header("Feature Importance Visualization")

# SHAP values can be slow to compute, so we compute them on the test set
# Also wrap in st.spinner in case it takes a moment
with st.spinner("Calculating SHAP values..."):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

col_shap1, col_shap2 = st.columns(2)
with col_shap1:
    st.subheader("SHAP Summary Plot")
    fig_shap, ax = plt.subplots(figsize=(8, 6))
    shap.summary_plot(shap_values, X_test, plot_type="dot", show=False)
    fig_shap.patch.set_facecolor('none')
    ax.set_facecolor('none')
    plt.tight_layout()
    st.pyplot(fig_shap, clear_figure=True)

with col_shap2:
    st.subheader("Feature Importance Bar Chart")
    fig_shap_bar, ax_bar = plt.subplots(figsize=(8, 6))
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
    fig_shap_bar.patch.set_facecolor('none')
    ax_bar.set_facecolor('none')
    plt.tight_layout()
    st.pyplot(fig_shap_bar, clear_figure=True)

# -----------------------------
# 4. RANDOM FOREST MODEL STRUCTURE
# -----------------------------
st.header("Random Forest Model Structure")

tree_id = st.slider("Select an Individual Tree to Explore", 0, n_trees - 1, 0)
st.write(f"Visualizing Tree #{tree_id} (max visual depth limited to 3 for readability)")

fig_tree, ax_tree = plt.subplots(figsize=(20, 8))
plot_tree(model.estimators_[tree_id], feature_names=selected_features, filled=True, max_depth=3, fontsize=10, ax=ax_tree, rounded=True, proportion=True)
fig_tree.patch.set_facecolor('none')
ax_tree.set_facecolor('none')
plt.tight_layout()
st.pyplot(fig_tree, clear_figure=True)

#streamlit run app.py

# -----------------------------
# MODEL UNDERSTANDING & IMPROVEMENT PROTOCOL
# -----------------------------
with st.expander("Model Understanding & Improvement Protocol (run analyses)", expanded=False):
    st.markdown("""
    This panel runs a structured analysis designed to understand what the model is really learning, expose weaknesses or biases,
    and produce concrete improvement suggestions. Steps performed when you press the button below:
    
    1) Compute feature importance and rank features by contribution.
    2) Analyze residuals (error distribution, residuals vs predictions, residuals over time).
    3) Compute correlations between residuals and input features to detect biases.
    4) Compare data distribution vs predictions to identify systematic mismatches.
    5) Produce non-trivial recommendations for feature engineering, model changes, and data cleaning.
    
    Every visualization answers a focused question (see captions).
    """)

    if st.button("Run Model Analysis"):
        try:
            # Basic checks
            if model is None:
                st.error("Model not available. Make sure the model was trained before running analysis.")
            else:
                residuals = (y_test - pred).reset_index(drop=True)
                y_test_idx = y_test.reset_index(drop=True)
                X_test_local = X_test.reset_index(drop=True)

                # 1) Feature importance (question: which features drive predictions?)
                st.subheader("Feature importance — which inputs most influence predictions?")
                try:
                    fi = getattr(model, 'feature_importances_', None)
                    if fi is None:
                        st.info("Feature importance not available for this model type.")
                    else:
                        fi_df = pd.DataFrame({ 'feature': X_test_local.columns, 'importance': fi })
                        fi_df = fi_df.sort_values('importance', ascending=False).reset_index(drop=True)
                        fig_fi = px.bar(fi_df, x='importance', y='feature', orientation='h', title='Feature importances (higher = more impact)', template='plotly_white')
                        st.plotly_chart(fig_fi, use_container_width=True)
                        st.markdown("Analytical question: Are the top features expected given domain knowledge? If not, investigate data leakage or proxy features.")

                except Exception as e:
                    st.error(f"Feature importance failed: {e}")

                # 2) Error distribution (question: how large and skewed are errors?)
                st.subheader("Error distribution — are errors biased or heavy-tailed?")
                fig_err_local = px.histogram(pd.DataFrame({'residuals': residuals}), x='residuals', nbins=50, title='Residual distribution', template='plotly_white')
                fig_err_local.update_layout(yaxis_title='count')
                st.plotly_chart(fig_err_local, use_container_width=True)

                q95 = np.percentile(np.abs(residuals), 95)
                st.write(f"95th percentile absolute error: {q95:.3f}")

                # 3) Residuals vs Predicted (question: are errors heteroscedastic or dependent on predicted magnitude?)
                st.subheader("Residuals vs Predicted — do errors grow with predicted value?")
                fig_rvp = px.scatter(x=pred, y=residuals, labels={'x':'Predicted','y':'Residual (Actual - Predicted)'}, trendline='lowess', title='Residuals vs Predicted', template='plotly_white')
                fig_rvp.add_hline(y=0, line_dash='dash', line_color='black')
                st.plotly_chart(fig_rvp, use_container_width=True)

                # 4) Residual correlation with features (question: which features explain residual patterns?)
                st.subheader("Residual correlation with input features — detect systematic biases")
                corr_list = []
                for col in X_test_local.columns:
                    try:
                        c = np.corrcoef(X_test_local[col].astype(float), residuals.astype(float))[0,1]
                    except Exception:
                        c = np.nan
                    corr_list.append((col, c))
                corr_df = pd.DataFrame(corr_list, columns=['feature','residual_corr']).dropna().sort_values('residual_corr', key=lambda s: s.abs(), ascending=False)
                if corr_df.empty:
                    st.info('No numeric correlation could be computed between residuals and features.')
                else:
                    st.dataframe(corr_df.head(10))
                    st.markdown("Analytical question: features with high residual correlation indicate model underfitting or missing interactions — consider adding feature transforms or interactions.")

                # 5) Residuals over time (question: are there temporal patterns?)
                if 'datetime' in df.columns:
                    st.subheader('Residuals over time — seasonality or drift?')
                    rtime_df = pd.DataFrame({'datetime': test_dates.reset_index(drop=True), 'residuals': residuals})
                    rtime_df = rtime_df.sort_values('datetime')
                    fig_rt = px.line(rtime_df, x='datetime', y='residuals', title='Residuals over time', template='plotly_white')
                    fig_rt.add_hline(y=0, line_dash='dash', line_color='black')
                    st.plotly_chart(fig_rt, use_container_width=True)

                # 6) Predictions vs Actual (question: where does the model systematically mispredict?)
                st.subheader('Predicted vs Actual — calibration and systematic offsets')
                pav = pd.DataFrame({'Actual': y_test_idx, 'Predicted': pred})
                fig_pa = px.scatter(pav, x='Actual', y='Predicted', trendline='ols', title='Predicted vs Actual', template='plotly_white')
                fig_pa.add_shape(type='line', x0=pav['Actual'].min(), y0=pav['Actual'].min(), x1=pav['Actual'].max(), y1=pav['Actual'].max(), line_dash='dash', line_color='black')
                st.plotly_chart(fig_pa, use_container_width=True)

                # 7) Data distribution vs predictions (question: do predictions occupy realistic ranges?)
                st.subheader('Distribution — Actual vs Predicted')
                dist_df = pd.DataFrame({'Actual': y_test_idx, 'Predicted': pred})
                fig_dist = go.Figure()
                fig_dist.add_trace(go.Histogram(x=dist_df['Actual'], name='Actual', opacity=0.7, nbinsx=50))
                fig_dist.add_trace(go.Histogram(x=dist_df['Predicted'], name='Predicted', opacity=0.7, nbinsx=50))
                fig_dist.update_layout(barmode='overlay', title='Actual vs Predicted distribution', template='plotly_white')
                st.plotly_chart(fig_dist, use_container_width=True)

                # Automated concise insights + recommendations (short, actionable)
                st.subheader('Concise insights & recommended next steps')
                insights = []
                # If top feature importance dominated by one feature, warn about proxies
                try:
                    if fi is not None:
                        top_ratio = fi_df['importance'].iloc[0] / fi_df['importance'].sum()
                        if top_ratio > 0.6:
                            insights.append('Single feature dominates importance (>{:.0%}). Check for proxies or leakage.'.format(top_ratio))
                except Exception:
                    pass

                # Residual correlation suggestions
                if not corr_df.empty:
                    strong = corr_df[corr_df['residual_corr'].abs() > 0.1]
                    if not strong.empty:
                        feats = strong['feature'].tolist()[:5]
                        insights.append(f"Residuals correlate with features: {', '.join(feats)}. Consider adding interactions, non-linear transforms, or feature-specific error models.")

                # Heteroscedasticity suggestion
                if np.corrcoef(np.abs(residuals), pred)[0,1] > 0.1:
                    insights.append('Errors increase with predicted magnitude — consider modeling variance (quantile regression) or transforming target (log).')

                # Data coverage warning
                if dist_df['Predicted'].max() > dist_df['Actual'].max() * 1.5:
                    insights.append('Predictions exceed historical observed range — check for extrapolation or sampling issues.')

                if insights:
                    for it in insights:
                        st.write('- ' + it)
                else:
                    st.write('No immediate, strong issues detected — focus on incremental feature engineering and validation.')

                st.markdown('---')
                st.markdown('Next concrete experiments you can run:')
                st.write('1) Add lag features and rolling means of top correlated features; 2) Try GradientBoostingRegressor or XGBoost and compare R²/MSE; 3) Run randomized hyperparameter search concentrating on max_depth and n_estimators; 4) Investigate temporal splits and seasonality features.')

        except Exception as e:
            st.error(f"Analysis failed: {e}")
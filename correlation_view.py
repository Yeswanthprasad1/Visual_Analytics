import streamlit as st
import pandas as pd
import numpy as np
import textwrap
import plotly.express as px
import plotly.graph_objects as go

@st.cache_data
def load_merged():
    path = "/Users/yeswanth/Desktop/VA/Dataset/Visual_Analytics/city_pollutant_health_merged_v2.csv"
    df = pd.read_csv(path)
    df["datetime"] = pd.to_datetime(df["datetime"]) if "datetime" in df.columns else pd.to_datetime(df[[c for c in df.columns if c.lower().startswith('date')][0]])
    return df

# Disease name mapping (same as used elsewhere)
DISEASE_NAME_MAPPING = {
    'TotaalNieuwvormingen_8': 'Total Neoplasms (Cancer)',
    'TotaalEndocrieneVoedingsStofwZ_32': 'Endocrine & Metabolic Diseases',
    'TotaalPsychischeStoornissen_35': 'Mental Disorders',
    'TotaalZiektenVanHartEnVaatstelsel_43': 'Cardiovascular Diseases (Total)',
    'TotaalZiektenVanDeKransvaten_44': 'Coronary Heart Diseases (Total)',
    'k_711AcuutHartinfarct_45': 'Acute Heart Infarction',
    'k_712OverigeZiektenVanDeKransvaten_46': 'Other Coronary Heart Diseases',
    'k_72OverigeHartziekten_47': 'Other Heart Diseases',
    'TotaalZiektenVanDeAdemhalingsorganen_50': 'Respiratory System Diseases (Total)',
    'k_81Griep_51': 'Influenza (Flu)',
    'k_82Longontsteking_52': 'Pneumonia',
    'TotaalChronischeAandOndersteLucht_53': 'Chronic Lower Respiratory Diseases',
    'k_831Astma_54': 'Asthma',
    'k_832OvChronAandOndersteLuchtw_55': 'Other Chronic Lower Respiratory',
    'k_84OverigeZiektenAdemhalingsorganen_56': 'Other Respiratory Diseases',
    'TotaalZiektenSpierenBeendBindwfsl_64': 'Musculoskeletal & Connective Tissue',
    'k_111ReumatoideArtritisEnArtrose_65': 'Rheumatoid Arthritis & Osteoarthritis',
    'k_112OvZktnSpierenBeendBindwfsl_66': 'Other Musculoskeletal'
}


def render_relation():
    st.title("Pollutant ⇄ Health Interactive Correlation Explorer")

    df = load_merged()
    # rename disease columns if present
    df.rename(columns=DISEASE_NAME_MAPPING, inplace=True)

    # Identify columns
    non_pollutant_cols = [c for c in ['datetime', 'City', 'Year'] if c in df.columns]
    disease_columns = [v for v in DISEASE_NAME_MAPPING.values() if v in df.columns]
    pollutants = [c for c in df.columns if c not in non_pollutant_cols + disease_columns]
    # Remove unwanted pollutant sensor columns from correlation analysis (case-insensitive)
    # include both 'so2' and 's02' variants to ensure removal
    exclude_set = {'s02', 'so2', 'co', 'nh3', 'h2s', 'ufp'}
    pollutants = [p for p in pollutants if p.lower() not in exclude_set]

    st.sidebar.header("Correlation Controls")

    # Pollutant and disease selectors
    default_polls = pollutants[:3] if len(pollutants) >= 3 else pollutants
    # prefer 'no2' and 'n02_palmes' if present (case-insensitive)
    preferred = []
    lower_map = {p.lower(): p for p in pollutants}
    for pref in ['no2', 'n02_palmes']:
        if pref in lower_map and lower_map[pref] not in preferred:
            preferred.append(lower_map[pref])
    if preferred:
        # fill remaining defaults with other pollutants
        for p in pollutants:
            if len(preferred) >= 3:
                break
            if p not in preferred:
                preferred.append(p)
        default_polls = preferred
    selected_pollutants = st.sidebar.multiselect("Select pollutants (x-axis features)", pollutants, default=default_polls)
    default_diseases = [d for d in ['Asthma', 'Chronic Lower Respiratory Diseases', 'Cardiovascular Diseases (Total)'] if d in disease_columns]
    selected_diseases = st.sidebar.multiselect("Select diseases (y-axis targets)", disease_columns, default=default_diseases)

    if not selected_pollutants or not selected_diseases:
        st.warning("Pick at least one pollutant and one disease to compute correlations.")
        return

    # Prepare analysis dataframe
    analysis_df = df[selected_pollutants + selected_diseases].dropna()
    if analysis_df.empty:
        st.error("No matching data after filtering. Adjust city/year selection or expand your selections.")
        return

    corr = analysis_df.corr().loc[selected_pollutants, selected_diseases]

    st.subheader("Correlation Heatmap")

    # Dynamically size the heatmap and wrap long axis labels so they tile when many items are selected
    n_p = len(selected_pollutants)
    n_d = len(selected_diseases)
    # adjust font size depending on number of labels
    axis_font_size = 12 if max(n_p, n_d) <= 10 else 10
    # height scales with the larger axis to keep cells readable
    heat_height = max(400, 30 * max(n_p, n_d))

    def wrap_label(s, width=18):
        # wrap on word boundaries and use <br> for Plotly multi-line tick labels
        if not isinstance(s, str):
            s = str(s)
        return '<br>'.join(textwrap.wrap(s, width=width))

    wrapped_x = [wrap_label(s, width=18) for s in selected_diseases]
    wrapped_y = [wrap_label(s, width=18) for s in selected_pollutants]

    # Use the raw correlation values but present wrapped tick labels for readability
    fig_heat = px.imshow(corr.values, x=selected_diseases, y=selected_pollutants,
                        color_continuous_scale='RdBu_r', zmin=-1, zmax=1,
                        labels=dict(x='Disease', y='Pollutant', color='Pearson r'))

    # Replace tick text with wrapped versions for tiling while keeping hover showing original names
    fig_heat.update_layout(height=heat_height,
                           xaxis=dict(ticktext=wrapped_x, tickvals=selected_diseases, tickangle=-45, tickfont=dict(size=axis_font_size)),
                           yaxis=dict(ticktext=wrapped_y, tickvals=selected_pollutants, tickfont=dict(size=axis_font_size)),
                           margin=dict(l=80, r=80, t=60, b=140))

    # Improve hovertemplate to show concise correlation
    fig_heat.update_traces(hovertemplate='Pollutant: %{y}<br>Disease: %{x}<br>Correlation: %{z:.3f}')

    st.plotly_chart(fig_heat, use_container_width=True)

    st.subheader("Grouped Correlation Bar Chart")
    melt_corr = corr.reset_index().melt(id_vars='index', var_name='Disease', value_name='Correlation').rename(columns={'index': 'Pollutant'})
    fig_bar = px.bar(melt_corr, x='Correlation', y='Disease', color='Pollutant', orientation='h', barmode='group', template='plotly_white')
    # scale bar chart height similarly so many labels don't overlap
    bar_height = max(400, 30 * max(n_p, n_d))
    fig_bar.update_layout(height=bar_height, margin=dict(l=80, r=40, t=40, b=80))
    st.plotly_chart(fig_bar, use_container_width=True)

    st.subheader("Scatter & Trend for Selected Pair")
    pair_pollutant = st.selectbox('Pollutant for scatter', selected_pollutants, index=0)
    pair_disease = st.selectbox('Disease for scatter', selected_diseases, index=0)

    scatter_df = df[[pair_pollutant, pair_disease]].dropna()
    if scatter_df.empty:
        st.error('No data for selected pair.')
        return

    fig_scatter = px.scatter(scatter_df, x=pair_pollutant, y=pair_disease, trendline='ols', template='plotly_white')
    st.plotly_chart(fig_scatter, use_container_width=True)

    # Display numeric correlation table
    st.subheader('Correlation Table')
    st.dataframe(corr)

    # Allow download of the correlation table
    csv = corr.to_csv().encode('utf-8')
    st.download_button('Download correlation CSV', data=csv, file_name='pollutant_disease_correlation.csv', mime='text/csv')

    st.markdown('---')
    st.markdown('Note: Correlation does not imply causation. Use this interactive tool to discover patterns and generate hypotheses for further study.')

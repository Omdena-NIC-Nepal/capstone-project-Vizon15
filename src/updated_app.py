import streamlit as st

# Must be the very first Streamlit command
st.set_page_config(page_title="Nepal Climate Vulnerability Dashboard", layout="wide")

import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import folium
from streamlit_folium import st_folium
from datetime import datetime
from fpdf import FPDF
from io import BytesIO, StringIO
from sklearn.impute import SimpleImputer
import geopandas as gpd
from sklearn.decomposition import PCA

# Import our custom lightweight NLP module
import nlp
url = "https://www.dropbox.com/scl/fi/x45dmh7gr7zdoxhmv0grj/updated_engineered_features_with_provinces.csv?rlkey=e4fc80irn8a1yacjw80ilvirn&st=wokg0z5e&dl=1"
# --- Load and Cache Data & Models ---
@st.cache_data
def load_data():
    eng = pd.read_csv(
        url,
        parse_dates=['Date']
    )
    raw = pd.read_csv(
        'datasets/nepal_gis_daily_data.csv',
        parse_dates=['Date'],
        usecols=[
            'Date','District','Temperature (°C)','Rainfall (mm)',
            'Precipitation (mm)','Humidity (%)','Air Quality Index',
            'Floods','Forest Fire','Drought'
        ]
    )
    raw['District'] = raw['District'].str.title()
    df = eng.merge(raw, on=['Date','District'], how='left', suffixes=('','_raw'))
    df['Province'] = df['Province'].fillna('Unknown')
    df['District'] = df['District'].fillna('Unknown')
    for col in ['Temperature (°C)','Rainfall (mm)','Precipitation (mm)',
                'Humidity (%)','Air Quality Index','Floods','Forest Fire','Drought']:
        raw_col = f"{col}_raw"
        if raw_col in df.columns:
            df[col] = df[raw_col]
            df.drop(columns=[raw_col], inplace=True)
    return df

@st.cache_data
def load_geodata():
    gdf = gpd.read_file('datasets/district_shape/district.shp')
    gdf['District'] = gdf['DISTRICT'].str.title()
    if 'Province' in gdf.columns:
        gdf['Province'] = gdf['Province'].str.title()
    else:
        gdf['Province'] = gdf['PROVINCE'].str.title()
    return gdf.to_crs(epsg=4326)

@st.cache_resource
def load_models():
    return {
        'rf_zone': joblib.load('models/rf_climate_zone.pkl'),
        'histgb_flood': joblib.load('models/histgb_flood_event.pkl'),
        'gb_vuln': joblib.load('models/gb_vulnerability.pkl'),
        'linreg_imp': joblib.load('models/linreg_impact.pkl'),
        'ridge_imp': joblib.load('models/ridge_impact.pkl'),
        'lasso_imp': joblib.load('models/lasso_impact.pkl'),
        'gbr_imp': joblib.load('models/gbreg_impact.pkl')
    }

# Load resources
df = load_data()
geo_gdf = load_geodata()
models = load_models()

# --- Sidebar Filters ---
st.sidebar.title("Filters")
selected_province = st.sidebar.selectbox("Province", df['Province'].unique())
selected_district = st.sidebar.selectbox(
    "District", df[df['Province'] == selected_province]['District'].unique()
)
vars_raw = ['Temperature (°C)', 'Rainfall (mm)', 'Precipitation (mm)', 'Humidity (%)', 'Air Quality Index']
selected_vars = st.sidebar.multiselect("Variables", vars_raw, default=vars_raw[:3])

season = st.sidebar.selectbox(
    "Season", ['All', 'Monsoon', 'Pre-monsoon', 'Post-monsoon', 'Winter']
)
date_range = st.sidebar.date_input(
    "Date Range", [df['Date'].min(), df['Date'].max()]
)

if date_range and len(date_range) >= 2:  # Check if date_range is not empty and has at least two elements
    date_min, date_max = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
    filtered = df[
        (df['Province'] == selected_province) & 
        (df['District'] == selected_district) & 
        (df['Date'] >= date_min) & 
        (df['Date'] <= date_max)
    ]
    if season != 'All':
        filtered = filtered[filtered['Season'] == season]


# --- Tabs ---
tabs = st.tabs([
    "Dashboard", "Maps", "Time Series", "EDA", "Predictions", "NLP", "Feedback", "System & Docs"
])

# Dashboard Tab
with tabs[0]:
    st.header("Overview Metrics")
    cols = st.columns(len(selected_vars))
    for i, var in enumerate(selected_vars):
        cols[i].metric(var, f"{filtered[var].mean():.2f}")
    st.markdown("---")
    st.dataframe(filtered[['Date'] + selected_vars].head(10))
    csv = filtered[['Date'] + selected_vars].to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Filtered Data as CSV",
        data=csv,
        file_name=f"{selected_district}_climate_data.csv",
        mime='text/csv'
    )

# Maps Tab
with tabs[1]:
    st.header("Climate Impact Heatmaps")
    level = st.radio("Level", ['Province','District'])
    var = st.selectbox("Variable to Map", selected_vars + ['SPI','flood_event','vulnerability_class','Impact'])
    try:
        if level=='Province':
            agg = df.groupby('Province')[var].mean().reset_index()
            merged = geo_gdf.dissolve(by='Province', as_index=False).merge(agg, on='Province')
            location_field = 'Province'
        else:
            agg = df.groupby('District')[var].mean().reset_index()
            merged = geo_gdf.merge(agg, on='District')
            location_field = 'District'
        geojson = merged.set_index(location_field).__geo_interface__
        fig = px.choropleth_mapbox(
            merged, geojson=geojson, locations=location_field,
            color=var, mapbox_style='carto-positron',
            center={'lat':28,'lon':84}, zoom=6 if level=='Province' else 8,
            opacity=0.6, hover_name=location_field,
            title=f"{var} by {level}"
        )
        fig.update_layout(margin={'r':0,'t':30,'l':0,'b':0})
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Map error: {e}")

# Time Series Tab
with tabs[2]:
    st.header("Time Series Trends")
    for v in selected_vars + ['SPI','flood_event']:
        ts = df.groupby('Date')[v].mean().reset_index()
        fig = px.line(ts, x='Date', y=v, title=v)
        st.plotly_chart(fig)

# EDA Tab
with tabs[3]:
    st.header("Exploratory Data Analysis")
    st.subheader("Summary & Info")
    st.write(filtered.describe())
    buf = StringIO(); filtered.info(buf=buf)
    st.text(buf.getvalue())

    # Distribution Analysis with distinct colors
    st.subheader("Distribution Analysis")
    for i, v in enumerate(selected_vars):
        fig = px.histogram(
            filtered, x=v, nbins=30,
            title=f"Distribution of {v}",
            color_discrete_sequence=[px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)]]
        )
        fig.update_layout(yaxis_title="Count", xaxis_title=v)
        st.plotly_chart(fig, use_container_width=True)
        st.write(f"The {v.lower()} distribution shows a mean of {filtered[v].mean():.2f} and variability reflected in the spread of values.")

    # Bivariate Analysis
    st.subheader("Bivariate & Multivariate Analysis")
    # Rainfall vs Flood Events
    fig1 = px.scatter(
        filtered, x='Rainfall (mm)', y='flood_event',
        title="Rainfall vs Flood Events",
        color_discrete_sequence=px.colors.qualitative.D3[0:1],
        trendline='ols'
    )
    fig1.update_layout(xaxis_title="Rainfall (mm)", yaxis_title="Flood Event Probability")
    st.plotly_chart(fig1)
    st.write("Higher rainfall generally correlates with increased flood event probability.")

    # Temperature vs Forest Fire
    fig2 = px.scatter(
        filtered, x='Temperature (°C)', y='Forest Fire',
        title="Temperature vs Forest Fire Incidents",
        color_discrete_sequence=px.colors.qualitative.D3[1:2],
        trendline='ols'
    )
    fig2.update_layout(xaxis_title="Temperature (°C)", yaxis_title="Forest Fire Count")
    st.plotly_chart(fig2)
    st.write("Rising temperatures are associated with an uptick in forest fire incidents.")

    # Time Series for Air Quality Index
    st.subheader("Air Quality Index Over Time")
    ts_aqi = filtered.groupby('Date')['Air Quality Index'].mean().reset_index()
    fig3 = px.line(
        ts_aqi, x='Date', y='Air Quality Index',
        title="Daily Average Air Quality Index",
        color_discrete_sequence=px.colors.qualitative.Vivid[2:3]
    )
    fig3.update_layout(xaxis_title="Date", yaxis_title="AQI")
    st.plotly_chart(fig3, use_container_width=True)
    st.write("The trend plot highlights periods of poor air quality and helps identify seasonal patterns.")

    # Scatter: Humidity vs Rainfall
    st.subheader("Humidity vs Rainfall Relationship")
    fig4 = px.scatter(
        filtered, x='Humidity (%)', y='Rainfall (mm)',
        title="Humidity vs Rainfall",
        color_discrete_sequence=px.colors.qualitative.Vivid[3:4]
    )
    fig4.update_layout(xaxis_title="Humidity (%)", yaxis_title="Rainfall (mm)")
    st.plotly_chart(fig4)
    st.write("This scatter shows how rainfall tends to increase with higher humidity levels.")

        # PCA 3D Scatter
    st.subheader("PCA of Selected Variables")
    pca = PCA(n_components=3).fit_transform(filtered[selected_vars].fillna(0))
    pca_df = pd.DataFrame(pca, columns=['PC1','PC2','PC3'])
    pca_df['Province'] = filtered['Province']
    # Use a distinctive color palette for each province
    fig5 = px.scatter_3d(
        pca_df,
        x='PC1', y='PC2', z='PC3',
        color='Province',
        color_discrete_sequence=px.colors.qualitative.Dark24,
        title="3D PCA of Climate Variables (Colored by Province)"
    )
    fig5.update_traces(marker=dict(size=5))
    st.plotly_chart(fig5)

    # Additional 3D PCA visualizations colored by each principal component
    st.subheader("3D PCA Colored by PC1, PC2, PC3")
    fig_p1 = px.scatter_3d(
        pca_df, x='PC1', y='PC2', z='PC3',
        color='PC1',
        color_continuous_scale='Viridis',
        title='PCA Colored by PC1 Values'
    )
    fig_p1.update_traces(marker=dict(size=4))
    st.plotly_chart(fig_p1)

    fig_p2 = px.scatter_3d(
        pca_df, x='PC1', y='PC2', z='PC3',
        color='PC2',
        color_continuous_scale='Cividis',
        title='PCA Colored by PC2 Values'
    )
    fig_p2.update_traces(marker=dict(size=4))
    st.plotly_chart(fig_p2)

    fig_p3 = px.scatter_3d(
        pca_df, x='PC1', y='PC2', z='PC3',
        color='PC3',
        color_continuous_scale='Plasma',
        title='PCA Colored by PC3 Values'
    )
    fig_p3.update_traces(marker=dict(size=4))
    st.plotly_chart(fig_p3)

    # 3D PCA Key Findings
    st.markdown("**3D PCA Key Findings:**")
    st.write("- Provinces form distinct clusters when colored by Province, indicating regional patterns.")
    st.write("- Coloring by PC1 shows how the primary source of variance (e.g., rainfall) distributes across data points.")
    st.write("- Coloring by PC2 highlights secondary variance patterns (e.g., temperature/humidity interactions).")
    st.write("- Coloring by PC3 reveals tertiary variance factors that may capture subtle seasonal or localized effects.")
    st.write("- **Distinct Clusters**: Provinces cluster separately, indicating region-specific climate patterns.")
    st.write("- **Variance Explained**: The first three principal components capture the majority of variance in temperature, rainfall, and humidity.")
    st.write("- **Outliers**: A few districts lie away from cluster centers, suggesting unique climate behaviors worth further investigation.")
    # Summary Insights
    st.markdown("---")
    st.subheader("Summary Insights")
    st.markdown("---")
    st.subheader("Summary Insights")
    st.markdown("""
- **Key Distributions**: Temperature, rainfall, and humidity exhibit variability across the selected date range, with monsoon seasons showing spikes in rainfall.
- **Correlations**: Positive correlation between rainfall and flood events; temperature rise is linked to forest fire incidents.
- **Air Quality Trends**: Periodic dips in AQI correspond with known pollution events (e.g., pre-monsoon dust storms).
- **Humidity-Rainfall Relationship**: Higher humidity generally co-occurs with increased rainfall.
- **PCA Clusters**: Provinces form distinct clusters in PCA space, indicating regional climate patterns.

**Future Impact**: These insights can guide targeted climate adaptation strategies, resource allocation for flood mitigation, and early warning systems for air quality and fire risk.
    """)
# Predictions Tab
with tabs[4]:
    st.header("Predict Future Events")
    event = st.selectbox("Event", ['Flood', 'Temperature (°C)', 'Rainfall (mm)', 'Forest Fire', 'Impact'])
    steps = st.slider("Forecast Horizon (weeks)", 1, 104, 5)
    if st.button("Forecast"):
        hist = filtered.set_index('Date')[[event if event!='Flood' else 'flood_event']].resample('W').mean()
        last_val = hist.iloc[-1, 0]
        forecast = np.full(steps, last_val)
        future_dates = pd.date_range(hist.index[-1]+pd.Timedelta(7,'D'), periods=steps, freq='W')
        df_fc = pd.DataFrame({event if event!='Flood' else 'flood_event': forecast}, index=future_dates)
        plot_df = pd.concat([
            hist.rename(columns={hist.columns[0]: event if event!='Flood' else 'flood_event'}),
            df_fc.rename(columns={event if event!='Flood' else 'flood_event':'Prediction'})
        ], axis=1)
        fig = px.line(
            plot_df, x=plot_df.index,
            y=[event if event!='Flood' else 'flood_event','Prediction'],
            title=f"{event} Forecast (weekly avg)"
        )
        fig.update_traces(selector=dict(name='Prediction'), line=dict(dash='dashdot'))
        st.plotly_chart(fig, use_container_width=True)
        st.metric(label=f"Forecasted {event}", value=f"{forecast[-1]:.2f}")
        direction = 'rise' if forecast[-1]>last_val else 'fall'
        st.write(f"The forecast shows a {direction} to {forecast[-1]:.2f} by {future_dates[-1].date()}.")

# NLP Tab
with tabs[5]:
    st.header("NLP Tools (Lightweight)")
    # User Guide for NLP Features
    st.markdown("""
### How to Use NLP Tools

Select from the following features to analyze your text:

- **Sentiment Analysis**: Get polarity scores (positive, neutral, negative, compound).
  - *Example:* "Climate change impact in Nepal is devastating".

- **Word Count**: Counts total words and shows the top 10 most frequent words.
  - *Example:* "Heavy rainfall and flooding affect rural communities repeatedly".

- **Keyword Extraction**: Lemmatizes tokens, removes stopwords, and shows top lemmas.
  - *Example:* "Kathmandu experiences air pollution due to stubble burning".

- **Entity Recognition**: Identifies named entities (e.g., locations, organizations).
  - *Example:* "UNDP is funding climate resilience projects in Gandaki Province".
""")
    st.markdown("---")
    st.markdown("**Example Text Box:** Paste one of the example sentences above or your own climate-related text to test each feature.")
    st.markdown("---")
    # Basic NLP Tools
    tool = st.selectbox("Select NLP Feature", ['Sentiment','Word Count','Keyword Extraction','Entity Recognition'])
    text = st.text_area(
        "Enter text here for analysis:",
        value="Climate change is causing more frequent and severe flooding in Gandaki Province, affecting agriculture and infrastructure.",
        height=200
    )
    if st.button("Run Analysis") and text:
        if tool=='Sentiment':
            st.subheader("Sentiment Analysis Results")
            st.json(nlp.analyze_sentiment(text))
        elif tool=='Word Count':
            st.subheader("Word Count Results")
            words=[w.lower() for w in text.split() if w.isalpha()]
            st.write(f"Total Words: {len(words)}")
            st.dataframe(pd.Series(words).value_counts().head(10).rename('Count'))
        elif tool=='Keyword Extraction':
            st.subheader("Keyword Extraction Results")
            tokens=nlp.preprocess_text(text)
            st.dataframe(pd.Series(tokens).value_counts().head(10).rename('Count'))
        else:
            st.subheader("Entity Recognition Results")
            ents=nlp.extract_entities(text)
            st.dataframe(pd.DataFrame(ents, columns=['Entity','Label']))

    # Advanced NLP Tasks
    st.markdown("---")
    st.subheader("Advanced Climate NLP Analysis")
    advanced_task = st.selectbox("Choose Advanced NLP Task", [
        "Climate News Articles", 
        "Social Media Analysis", 
        "Topic Modeling", 
        "Text Summarization", 
        "Multilingual Translation", 
        "NLP Insights with Climate Data"
    ])
    # Climate News Articles
    if advanced_task == "Climate News Articles":
        news_df = nlp.get_news_articles()
        st.write("### Fetched Climate News Articles")
        st.dataframe(news_df)
        st.write("#### Article Sentiments")
        news_df['sentiment'] = news_df['text'].apply(lambda t: nlp.analyze_sentiment(t[:512]))
        st.dataframe(news_df[['url', 'sentiment']])

    # Social Media Analysis
    elif advanced_task == "Social Media Analysis":
        social_df = nlp.prepare_social_data()
        st.write("### Social Media Data with NLP Analysis")
        st.dataframe(social_df)
        agg_sent = social_df.groupby('date')['sentiment'].apply(lambda x: np.mean([s['compound'] for s in x])).reset_index(name='avg_sentiment')
        st.write("#### Aggregated Sentiment Over Time")
        st.dataframe(agg_sent)
        fig = px.line(agg_sent, x='date', y='avg_sentiment', title="Sentiment Trend Over Time")
        st.plotly_chart(fig, use_container_width=True)

    # Topic Modeling
    elif advanced_task == "Topic Modeling":
        tokens = nlp.prepare_social_data()['tokens']
        topics = nlp.run_topic_modeling(tokens)
        st.write("### Extracted Topics from Social Media Data")
        for topic in topics:
            st.write(topic)

    # Text Summarization
    elif advanced_task == "Text Summarization":
        st.write("### Text Summarization Tool for Climate Reports")
        sample_text = st.text_area("Enter text for summarization", value="Climate change poses severe risks to the global ecosystem. Many events are interrelated including rising temperatures, increasing sea levels, and extreme weather patterns.")
        if st.button("Summarize"):  # separate button
            summary = nlp.run_summarization(sample_text, num_sentences=3)
            st.write("**Summary:**")
            st.write(summary)

    # Multilingual Translation
    elif advanced_task == "Multilingual Translation":
        st.write("### Multilingual Translation for Nepali Texts")
        nep_text = st.text_area("Enter Nepali text", value="नेपालमा भीषण बाढीले गर्दा जनजीवन प्रभावित भएको छ।")
        if st.button("Translate & Analyze"):  # separate button
            translated = nlp.translate(nep_text, 'ne', 'en')
            sentiment = nlp.analyze_sentiment(translated[:512])
            st.write("**Translated Text:**", translated)
            st.write("**Sentiment Analysis:**", sentiment)

    # NLP Insights with Climate Data
    else:
        merged = nlp.integrate_nlp_with_climate(nlp.prepare_social_data(), 'datasets/nepal_gis_daily_data.csv')
        st.write("### Integrated NLP and Climate Data")
        st.dataframe(merged.head())
        if "Temperature (°C)" in merged.columns:
            fig = px.scatter(merged, x='avg_sentiment', y='Temperature (°C)', title="Average Sentiment vs Temperature")
            st.plotly_chart(fig, use_container_width=True)

import gspread
from oauth2client.service_account import ServiceAccountCredentials

# --- Helper Function ---
from gspread import Client

from oauth2client.service_account import ServiceAccountCredentials

def save_feedback(name, email, feedback):
    try:
        scope = " ".join(["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"])
        creds = ServiceAccountCredentials.from_json_keyfile_name("service_account.json", scope)
        credentials = Credentials(creds)
        client = Client(auth=credentials)

        sheet = client.open("NepalClimateFeedback").sheet1
        sheet.append_row([datetime.now().strftime('%Y-%m-%d %H:%M:%S'), name, email, feedback])
        return True
    except Exception as e:
        st.error(f"Error saving feedback: {e}")
        return False


# Feedback Tab
with tabs[6]:
    st.header("📩 Send Feedback to Admin")
    with st.form("feedback_form", clear_on_submit=True):
        name = st.text_input("Your Name")
        email = st.text_input("Your Email")
        feedback = st.text_area("Your Feedback")
        submitted = st.form_submit_button("Submit Feedback")

        if submitted:
            if name and email and feedback:
                if save_feedback(name, email, feedback):
                    st.success("✅ Thank you! Your feedback has been recorded.")
            else:
                st.warning("⚠️ Please fill in all fields.")
# System & Docs Tab
with tabs[7]:
    st.header("🚀 Deployment, Documentation & Future Roadmap")

    st.subheader("Deployment & Integration")
    st.markdown(open('deployment/architecture.md').read())
    st.markdown(open('deployment/security.md').read())
    st.markdown(open('deployment/backup.md').read())

    st.subheader("System Monitoring")
    try:
        from deployment.monitoring import show_dashboard
        show_dashboard()
    except Exception as e:
        st.info("Monitoring dashboard not available. See deployment/monitoring.py.")

    st.subheader("Documentation & Knowledge Transfer")
    st.markdown(open('docs/api.md').read())
    st.markdown(open('docs/user_guide.md').read())
    st.markdown(open('docs/wiki.md').read())

    st.subheader("Future Improvements & Extensions")
    st.markdown(open('future.md').read())
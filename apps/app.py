import streamlit as st
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

# Ensure PCA is imported for multivariate analysis
# Light-weight NLP libraries
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
nltk.download('stopwords')
nltk.download('vader_lexicon')
from nltk.corpus import stopwords

# --- App Configuration ---
st.set_page_config(page_title="Nepal Climate Vulnerability Dashboard", layout="wide")

# --- Load and Cache Data & Models ---
@st.cache_data
def load_data():
    eng = pd.read_csv('/Users/vizon/github-classroom/Omdena-NIC-Nepal/capstone-project-Vizon15/datasets/processed/updated_engineered_features_with_provinces.csv', parse_dates=['Date'])
    raw = pd.read_csv(
        '/Users/vizon/github-classroom/Omdena-NIC-Nepal/capstone-project-Vizon15/datasets/nepal_gis_daily_data.csv', parse_dates=['Date'],
        usecols=['Date','District','Temperature (°C)','Rainfall (mm)','Precipitation (mm)','Humidity (%)','Air Quality Index','Floods','Forest Fire','Drought']
    )
    raw['District'] = raw['District'].str.title()
    # Merge with suffix for raw columns
    df = eng.merge(
        raw, on=['Date','District'], how='left', suffixes=('','_raw')
    )
    df['Province'] = df['Province'].fillna('Unknown')
    df['District'] = df['District'].fillna('Unknown')
    # Overwrite base columns with raw for user clarity
    for col in ['Temperature (°C)','Rainfall (mm)','Precipitation (mm)','Humidity (%)','Air Quality Index','Floods','Forest Fire','Drought']:
        raw_col = f"{col}_raw"
        if raw_col in df.columns:
            df[col] = df[raw_col]
            df.drop(columns=[raw_col], inplace=True)
    return df

@st.cache_data
def load_geodata():
    gdf = gpd.read_file('/Users/vizon/github-classroom/Omdena-NIC-Nepal/capstone-project-Vizon15/datasets/district_shape/district.shp')
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
selected_district = st.sidebar.selectbox("District", df[df['Province']==selected_province]['District'].unique())
vars_raw = ['Temperature (°C)','Rainfall (mm)','Precipitation (mm)','Humidity (%)','Air Quality Index']
selected_vars = st.sidebar.multiselect("Variables", vars_raw, default=vars_raw[:3])
season = st.sidebar.selectbox("Season", ['All','Monsoon','Pre-monsoon','Post-monsoon','Winter'])
date_range = st.sidebar.date_input("Date Range", [df['Date'].min(), df['Date'].max()])
date_min = pd.to_datetime(date_range[0])
date_max = pd.to_datetime(date_range[1])
filtered = df[(df['Province']==selected_province)&(df['District']==selected_district)&(df['Date']>=date_min)&(df['Date']<=date_max)]
if season!='All': filtered = filtered[filtered['Season']==season]

# --- Tabs ---
tabs = st.tabs(["Dashboard","Maps","Time Series","EDA","Predictions","NLP","Feedback"])

# Dashboard
with tabs[0]:
    st.header("Overview Metrics")
    cols = st.columns(len(selected_vars))
    for i,var in enumerate(selected_vars): cols[i].metric(var, f"{filtered[var].mean():.2f}")
    st.markdown("---")
    st.dataframe(filtered[['Date'] + selected_vars].head(10))
        # Download filtered data
    csv = filtered[['Date'] + selected_vars].to_csv(index=False).encode('utf-8')
    st.download_button(
            label="📥 Download Filtered Data as CSV",
            data=csv,
            file_name=f"{selected_district}_climate_data.csv",
            mime='text/csv'
        )

# Maps
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
        # Use Plotly Choropleth for interactivity
        geojson = merged.set_index(location_field).__geo_interface__
        fig = px.choropleth_mapbox(
            merged,
            geojson=geojson,
            locations=location_field,
            color=var,
            mapbox_style='carto-positron',
            center={'lat':28,'lon':84},
            zoom=6 if level=='Province' else 8,
            opacity=0.6,
            hover_name=location_field,
            title=f"{var} by {level}"
        )
        fig.update_layout(margin={'r':0,'t':30,'l':0,'b':0})
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Map error: {e}")

# Time Series
with tabs[2]:
    st.header("Time Series Trends")
    for v in selected_vars+['SPI','flood_event']:
        ts = df.groupby('Date')[v].mean().reset_index()
        fig = px.line(ts, x='Date', y=v, title=v)
        st.plotly_chart(fig)

# EDA
with tabs[3]:
    st.header("Exploratory Data Analysis")
    st.subheader("Summary & Info")
    st.write(filtered.describe())
    buf = StringIO()
    filtered.info(buf=buf)
    info_df = buf.getvalue()
    st.text(info_df)
    st.subheader("Distribution Analysis")
    for v in selected_vars:
        st.plotly_chart(px.histogram(filtered, x=v, nbins=30))
    st.subheader("Bivariate & Multivariate")
    fig = px.scatter(filtered, x='Rainfall (mm)', y='flood_event', trendline='ols')
    st.plotly_chart(fig)
    fig = px.scatter(filtered, x='Temperature (°C)', y='Forest Fire', trendline='ols')
    st.plotly_chart(fig)
    from sklearn.decomposition import PCA
    pca = PCA(n_components=3).fit_transform(filtered[selected_vars].fillna(0))
    pca_df = pd.DataFrame(pca, columns=['PC1','PC2','PC3'])
    pca_df['Province']=filtered['Province']
    st.plotly_chart(px.scatter_3d(pca_df, x='PC1', y='PC2', z='PC3', color='Province'))

# Predictions
with tabs[4]:
    st.header("Predict Future Events")
    event = st.selectbox("Event", ['Flood','Temperature (°C)','Rainfall (mm)','Forest Fire','Impact'])
    steps = st.slider("Forecast Horizon (weeks)",1,104,5)
    if st.button("Forecast"):
        # Weekly resample for clarity
        hist = filtered.set_index('Date')[[event if event!='Flood' else 'flood_event']].resample('W').mean()
        last_val = hist.iloc[-1,0]
        forecast = np.full(steps, last_val)
        future_dates = pd.date_range(hist.index[-1] + pd.Timedelta(7,'D'), periods=steps, freq='W')
        df_fc = pd.DataFrame({event if event!='Flood' else 'flood_event': forecast}, index=future_dates)
        plot_df = pd.concat([hist.rename(columns={hist.columns[0]:event if event!='Flood' else 'flood_event'}), df_fc.rename(columns={event if event!='Flood' else 'flood_event':'Prediction'})], axis=1)
        fig = px.line(plot_df, x=plot_df.index, y=[event if event!='Flood' else 'flood_event','Prediction'], title=f"{event} Forecast (weekly avg)")
        fig.update_traces(selector=dict(name='Prediction'), line=dict(color='red', width=4, dash='dashdot'))
        fig.update_traces(selector=dict(name=event if event!='Flood' else 'flood_event'), line=dict(color='blue', width=2))
        # Limit ticks for readability
        if len(plot_df) > 20:
            fig.update_xaxes(nticks=20)
        st.plotly_chart(fig, use_container_width=True)
        # Display forecast value and guidance
        pred_val = forecast[-1]
        st.metric(label=f"Forecasted {event}", value=f"{pred_val:.2f}")
        direction = 'rise' if pred_val > last_val else 'fall'
        st.write(f"The forecast shows a {direction} to {pred_val:.2f} by {future_dates[-1].date()}, indicating potential change in {event.lower()} trends.")

# NLP
with tabs[5]:
    st.header("NLP Tools (Lightweight)")
    st.markdown(
        """
**NLP Tools Guide:**

Use these tools to analyze text data related to climate. You can:
- **Sentiment**: Understand overall sentiment (positive/negative/neutral).
- **Word Count**: Check total words and most frequent words.
- **Keyword Extract**: Identify top keywords by frequency.
- **POS Tagging**: View parts-of-speech distribution.

**Example Text:**
> Climate change is causing more frequent and severe flooding in Gandaki Province, affecting agriculture and infrastructure.
        """
    )
    tool = st.selectbox("Tool", ['Sentiment','Word Count','Keyword Extract','POS Tagging'])
    text = st.text_area(
        "Enter text here...",
        value="Climate change is causing more frequent and severe flooding in Gandaki Province, affecting agriculture and infrastructure.",
        height=200,
        placeholder="Type or paste your text here for analysis..."
    )
    if st.button("Run") and text:
        if tool == 'Sentiment':
            sid = SentimentIntensityAnalyzer()
            scores = sid.polarity_scores(text)
            st.write("**Sentiment Scores:**")
            st.json(scores)
        elif tool == 'Word Count':
            words = [w.lower() for w in text.split() if w.isalpha()]
            total = len(words)
            freq = pd.Series(words).value_counts().head(10)
            st.write(f"**Total Words:** {total}")
            st.write("**Top 10 Words:**")
            st.dataframe(freq.rename('Count'))
        elif tool == 'Keyword Extract':
            # Remove stopwords
            sw = set(stopwords.words('english'))
            tokens = [w.lower() for w in text.split() if w.isalpha()]
            keywords = [w for w in tokens if w not in sw]
            freq = pd.Series(keywords).value_counts().head(10)
            st.write("**Top 10 Keywords (stopwords removed):**")
            st.dataframe(freq.rename('Count'))
        else:
            # POS Tagging
            import nltk as _nltk
            _nltk.download('averaged_perceptron_tagger')
            tokens = [w for w in text.split() if w.isalpha()]
            tags = _nltk.pos_tag(tokens)
            pos_df = pd.DataFrame(tags, columns=['Word','POS'])
            st.write("**Part-of-Speech Tags:**")
            st.dataframe(pos_df)
            # POS distribution
            dist = pos_df['POS'].value_counts()
            fig = px.bar(dist.reset_index(), x='index', y='POS', labels={'index':'POS','POS':'Count'}, title='POS Tag Distribution')
            st.plotly_chart(fig, use_container_width=True)

# Feedback
with tabs[6]:
    st.header("Feedback")
    fb=st.text_area("Feedback")
    if st.button("Submit"): open('feedback.csv','a').write(f"{datetime.now()},{fb}\n")
    st.success("Thanks!")

st.markdown("*©2025 Nepal Climate App*.*")

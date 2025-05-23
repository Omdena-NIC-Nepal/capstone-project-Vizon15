# Nepal Climate Vulnerability Analysis & Dashboard

A comprehensive geospatial and machine learning project for climate vulnerability assessment across Nepal. The project features deep Exploratory Data Analysis (EDA), robust predictive modeling, NLP-driven climate insights, and a richly interactive Streamlit dashboard designed for both technical and non-technical audiences.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Quick Start](#quick-start)
- [Data Sources & Processing](#data-sources--processing)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Model Development](#model-development)
- [NLP Integration](#nlp-integration)
- [Dashboard Application](#dashboard-application)
- [Usage & Interactions](#usage--interactions)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

---

**Project URL:** [Capstone Project](https://capstone-project-vizon15.streamlit.app/)

## 🛠️ Technologies Used

- **Python**: Primary programming language
- **Streamlit**: Web application framework
- **Google Sheets API**: Data storage & feedback integration
- **Plotly & Folium**: GIS mapping & visualization
- **gspread**: Google Sheets API wrapper
- **OAuth 2.0**: Secure authentication for Google services

## 📥 How to Use

1. Visit the [Capstone Project](https://capstone-project-vizon15.streamlit.app/)
2. Interact with the GIS maps and analyze climate trends.
3. Submit feedback directly within the app.
4. Developers can contribute by improving data analysis workflows.

## 📌 Overview

This project is designed to provide advanced Geographic Information System (GIS) data analysis, focusing on Nepal's climate and environmental insights. Built using **Streamlit**, the application enables users to visualize spatial data, analyze geospatial trends, and collect feedback to improve climate-related research.

- **Comprehensive EDA** of daily climate, hazard, and impact data.
- **Geospatial mapping** of climate variables, hazards, and vulnerability indices by district/province.
- **Advanced ML models** to forecast floods, droughts, fire risks, and impacts.
- **NLP-powered insights** from news and social media.
- **An interactive Streamlit dashboard** for real-time analysis, visualization, and reporting.

---

## Features

- 📈 **Geospatial EDA** (maps, time series, dynamic plots)
- 🏔 **Elevation-based climate analysis** (e.g., temperature trends by elevation bands)
- ☔ **Event frequency mapping** (flood, drought, fire, glacial lakes)
- 📊 **Correlation & trend analysis** (including Mann–Kendall, regression)
- 🌍 **Choropleth maps** (static and interactive, with time sliders)
- ⚡ **Machine Learning**: classification (Random Forests, Gradient Boosting), regression (Linear, Ridge, Lasso, GB)
- 🧠 **NLP**: sentiment, topic modeling, entity recognition, news/social scraping
- 🌐 **Streamlit dashboard**: filters, maps, time series, model predictions, PDF downloads
- 📤 **Integrated feedback system** and real-time documentation
- **Interactive Maps**: Visualize GIS data with choropleth and custom map layers.
- **Climate Data Analysis**: Analyze temperature, precipitation, and environmental patterns.
- **Google Sheets Integration**: Collect and manage user feedback dynamically.
- **Secure Secrets Management**: Implements best practices for handling API keys and credentials.
- **Streamlit-Powered UI**: User-friendly and efficient interface for seamless data exploration.

---

## Repository Structure

```
|capstone-project-Vizon15
├── ./ASSIGNMENT.md
├── ./apps
│   └── ./apps/app.py
├── ./datasets
│   ├── ./datasets/district_shape
│   │   ├── ./datasets/district_shape/district.cpg
│   │   ├── ./datasets/district_shape/district.dbf
│   │   ├── ./datasets/district_shape/district.geojson
│   │   ├── ./datasets/district_shape/district.prj
│   │   ├── ./datasets/district_shape/district.shp
│   │   └── ./datasets/district_shape/district.shx
│   ├── ./datasets/nepal_gis_daily_data.csv
│   ├── ./datasets/processed
│   │   ├── ./datasets/processed/preprocessed_climate_data.csv
│   │   └── ./datasets/processed/updated_engineered_features_with_provinces.csv
│   └── ./datasets/synthetic_district_elevation.csv
├── ./deployment
│   ├── ./deployment/architecture.md
│   ├── ./deployment/backup.md
│   ├── ./deployment/ci_cd.yml
│   ├── ./deployment/monitoring.py
│   ├── ./deployment/pipeline.py
│   └── ./deployment/security.md
├── ./docs
│   ├── ./docs/api.md
│   ├── ./docs/code_comments.md
│   ├── ./docs/faq.md
│   ├── ./docs/maintenance.md
│   ├── ./docs/training.md
│   ├── ./docs/tutorials.md
│   ├── ./docs/update_procedure.md
│   ├── ./docs/user_guide.md
│   └── ./docs/wiki.md
├── ./future.md
├── ./models
│   ├── ./models/gb_vulnerability.pkl
│   ├── ./models/gbreg_impact.pkl
│   ├── ./models/histgb_flood_event.pkl
│   ├── ./models/lasso_impact.pkl
│   ├── ./models/linreg_impact.pkl
│   ├── ./models/rf_climate_zone.pkl
│   └── ./models/ridge_impact.pkl
├── ./output
├── ./requirements.txt
├── ./src
│   ├── ./src/__pycache__
│   │   └── ./src/__pycache__/nlp.cpython-311.pyc
│   ├── ./src/app.py
│   ├── ./src/data_gen.py
│   ├── ./src/data_preprocessing.ipynb
│   ├── ./src/eda.ipynb
│   ├── ./src/feature_engineering.py
│   ├── ./src/model_dev.py
│   ├── ./src/model_validation.py
│   ├── ./src/nlp.py
│   ├── ./src/test.ipynb
│   ├── ./src/updated_app.py
│   └── ./src/updated_nlp.py
├── ./README.md
└── ./repo_structure.txt

```

---

## Getting Started

### Prerequisites

- Python 3.8+
- `pip` or `conda` for package management

### Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/Omdena-NIC-Nepal/capstone-project-Vizon15.git
   cd capstone-project-Vizon15
   ```

2. **Create & activate a virtual environment**

   ```bash
   python -m venv venv
   source venv/bin/activate       # Linux/macOS
   venv\Scripts\activate          # Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Quick Start

- **Run the dashboard:**
  ```bash
  streamlit run app.py
  ```
- **App will open in your browser at** `http://localhost:8501`

---

## Data Sources & Processing

- **Climate Data**: Daily records with temperature, precipitation, humidity, flood/drought/fire counts, glacial lake counts, yield/loss, population exposure.
- **District Shapefile**: GIS boundaries for all 77 districts.
- **Elevation Data**: Synthetic or measured average elevation per district.
- **Automated loading**: Large datasets are streamed from Dropbox to avoid GitHub file limits.

**Raw data is processed with:**

- Imputation (SimpleImputer/KNN)
- Feature engineering (date parts, lag variables, province/district labels)
- Scaling (MinMaxScaler)
- Integration (merging engineered features with raw climate & impact data)
- All processing code is in `data_preprocessing.ipynb` and `eda.ipynb`.

**Example code for dynamic data loading:**

```python
url = "https://www.dropbox.com/scl/fi/x45dmh7gr7zdoxhmv0grj/updated_engineered_features_with_provinces.csv?rlkey=e4fc80irn8a1yacjw80ilvirn&st=wokg0z5e&dl=1"
df = pd.read_csv(url)
```

---

## Exploratory Data Analysis (EDA)

See `eda.ipynb` for in-depth analysis and visualizations:

- **Temperature trends** by province, district, elevation band
- **Monthly precipitation** patterns and year-on-year trends
- **Extreme weather** frequency: annual top 5% events (floods, droughts, fires)
- **Glacial retreat**: glacial lake formation trends
- **Correlation heatmaps**: climate variables vs. agricultural yield, loss, population exposure
- **Vulnerability mapping**: composite indices, time-slider choropleths
- **Statistical tests**: Mann–Kendall trend, linear regression

---

## Model Development

Automated model training and evaluation in `model_dev.py`:

- **Data loading**: from `output/engineered_features.csv`
- **Imputation & splitting**: handles missing values, train/test splits
- **Algorithms:**
  - Classification: Random Forest, Gradient Boosting, HistGradientBoosting
  - Regression: Linear, Ridge, Lasso, GradientBoostingRegressor
- **Cross-validation**: KFold, cross_val_score
- **Metrics**: Accuracy, F1, MSE, MAE, R²
- **Model caching**: skips retraining if `.pkl` artifacts exist

**Usage:**

```bash
python model_dev.py
```

---

## NLP Integration

NLP tools in `nlp.py` provide climate-driven insights:

- **Web scraping**: Fetch news/social posts (requests, BeautifulSoup)
- **Preprocessing**: Regex, lemmatization, stopwords
- **Sentiment analysis**: VADER
- **Topic modeling**: LDA
- **Named Entity Recognition**: Regex-based, simple NER
- **Data synthesis**: Generate synthetic posts with location/timestamp
- **Integration**: Merge sentiment with daily climate data
- **Caching**: via Streamlit `@st.cache_data`

---

## Dashboard Application

The interactive dashboard (`app.py`) offers:

- **Sidebar filters**: Province, district, variables, season, date range
- **Tabs for**:
  - Dashboard: Overview metrics, data preview, CSV download
  - Maps: Interactive heatmaps at district/province level
  - Time Series: Trends for any variable over time
  - EDA: Summary stats, distributions, bivariate/multivariate plots, PCA
  - Predictions: ML-based forecasts; supports file upload for batch inference
  - NLP: Sentiment, keywords, NER, advanced analysis on custom or fetched text
  - Feedback: Send feedback (stored in Google Sheets, GCP credentials required)
  - System & Docs: Deployment, monitoring, API/user/wiki docs, roadmap

**All features update dynamically based on sidebar filter selections.**

---

## Usage & Interactions

### Typical User Flow

1. **Launch the dashboard**
   - `streamlit run app.py`
2. **Apply filters** (province, district, date, variables)
3. **Explore**
   - Metrics and tables (Dashboard tab)
   - Maps (Maps tab)
   - Time series and trends (Time Series tab)
   - EDA, PCA, and insights (EDA tab)
4. **Forecast events** (Predictions tab)
   - Select event type, forecast horizon
   - Optionally upload CSV for batch predictions (see template/column requirements in the UI)
5. **Analyze text** (NLP tab)
   - Run sentiment, entity, or topic analysis on any provided or fetched text
   - Use advanced tools for news/social media, translation, summarization
6. **Send feedback** (Feedback tab)
7. **View documentation and system info** (System & Docs tab)

### Data Access

- All large datasets are automatically streamed from Dropbox—**no manual download needed**.
- To change the data source, update the Dropbox link in code.

---

## Contributing

We welcome contributions!

1. **Fork this repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. **Commit your changes**
   ```bash
   git commit -m "Add new feature: ..."
   ```
4. **Push and open a Pull Request**
   ```bash
   git push origin feature/your-feature-name
   ```
5. **Describe your changes clearly**; adhere to code style and add tests or notebook examples for new functionality.

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## Contact

- **Lead:** Jhalak Bahadur Khatri
- **Email:** jhalakkc5@gmail.com
- **GitHub:** [Vizon15](https://github.com/Vizon15)
- **Project Org:** [Omdena NIC Nepal](https://omdena.com/)

---

_For questions, suggestions, or issues, open an [issue](https://github.com/Omdena-NIC-Nepal/capstone-project-Vizon15/issues) or use the dashboard’s Feedback tab!_

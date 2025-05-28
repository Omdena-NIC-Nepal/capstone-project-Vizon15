# feature_engineering.py
"""
Optimized Feature Engineering for Nepal Climate GIS Data
- Selective column loading to reduce memory
- Efficient imputation, normalization, lag features via rolling
- Vectorized climate indices and zone assignment
- Incremental PCA for dimensionality reduction
- Creation of flood and vulnerability targets
"""
import os
import pandas as pd
import numpy as np
import geopandas as gpd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import IncrementalPCA

# Configuration
RAW_DATA_PATH = '/Users/vizon/github-classroom/Omdena-NIC-Nepal/capstone-project-Vizon15/datasets/nepal_gis_daily_data.csv'
SHAPE_PATH = '/Users/vizon/github-classroom/Omdena-NIC-Nepal/capstone-project-Vizon15/datasets/district_shape/district.shp'
ELEV_PATH = '/Users/vizon/github-classroom/Omdena-NIC-Nepal/capstone-project-Vizon15/datasets/synthetic_district_elevation.csv'
OUTPUT_PATH = 'output'
ENGINEERED_FILE = f'{OUTPUT_PATH}/engineered_features.csv'

# 1. Load only necessary columns
usecols = [
    'Date','District','Rainfall (mm)','Temperature (°C)','Humidity (%)',
    'Precipitation (mm)','Floods','Drought','Forest Fire',
    'Air Quality Index','Infrastructure Risk (damage level)',
    'Population Exposure (count)'
]
climate_df = pd.read_csv(
    RAW_DATA_PATH,
    parse_dates=['Date'],
    usecols=usecols,
    dtype={'District':'category'}
)
# Load GIS shapes
districts_gdf = gpd.read_file(SHAPE_PATH)
districts_gdf['District'] = districts_gdf['DISTRICT'].str.title().astype('category')

# 2. Basic Imputation and Indices
num_cols = ['Rainfall (mm)','Temperature (°C)','Precipitation (mm)']
imp_mean = SimpleImputer(strategy='mean')
climate_df[num_cols] = imp_mean.fit_transform(climate_df[num_cols])
# SPI & HSI
prec = climate_df['Precipitation (mm)']
climate_df['SPI'] = (prec - prec.mean()) / prec.std()
climate_df['HSI'] = climate_df['Temperature (°C)'] * climate_df['Humidity (%)'] / 100

# 3. Seasonal Flags
months = climate_df['Date'].dt.month
climate_df['Monsoon'] = months.between(6,9).astype('uint8')
season_map = {1:'Winter',2:'Winter',3:'Pre-monsoon',4:'Pre-monsoon',5:'Pre-monsoon',
              6:'Monsoon',7:'Monsoon',8:'Monsoon',9:'Monsoon',
              10:'Post-monsoon',11:'Post-monsoon',12:'Winter'}
climate_df['Season'] = months.map(season_map).astype('category')

# 4. Lag Features via Rolling Means
climate_df.sort_values(['District','Date'], inplace=True)
for window in [1,3,7,30]:
    # Explicit observed=False to match current pandas behavior
    rain_grp = climate_df.groupby('District', observed=False)['Rainfall (mm)']
    temp_grp = climate_df.groupby('District', observed=False)['Temperature (°C)']

    climate_df[f'Rain_lag_{window}'] = (
        rain_grp
        .shift(1)
        .rolling(window, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )
    climate_df[f'Temp_lag_{window}'] = (
        temp_grp
        .shift(1)
        .rolling(window, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )

# 5. Spatial Proximity and Density
# Reproject to a metric CRS for accurate spatial calculations
# Choose a suitable projected CRS, e.g., UTM zone for Nepal (EPSG:32645)
districts_proj = districts_gdf.to_crs(epsg=32645)
# Compute centroids in projected CRS
districts_proj['centroid'] = districts_proj.geometry.centroid
# Extract Kathmandu centroid
capital = districts_proj.loc[districts_proj['District']=='Kathmandu','centroid'].iloc[0]
# Distance to capital
districts_proj['Dist_to_Capital'] = districts_proj['centroid'].distance(capital)
# Area in square meters
districts_proj['Area'] = districts_proj.geometry.area
# Merge back spatial columns
dist_info = districts_proj[['District','Dist_to_Capital','Area']]
climate_df = climate_df.merge(
    dist_info,
    on='District', how='left'
)
# Population density per area (count per m^2)
climate_df['Pop_Density'] = climate_df['Population Exposure (count)'] / climate_df['Area']

# 6. Vulnerability Normalization & One-Hot Vulnerability Normalization & One-Hot
vuln_cols = ['Floods','Drought','Forest Fire','Air Quality Index']
scaler_vuln = MinMaxScaler()
climate_df[[f'{c}_norm' for c in vuln_cols]] = (
    scaler_vuln.fit_transform(climate_df[vuln_cols])
)
climate_df = pd.get_dummies(
    climate_df,
    columns=['Infrastructure Risk (damage level)'],
    prefix='InfraRisk'
)

# 7. Scaling for PCA
scale_cols = [
    'Rainfall (mm)','Temperature (°C)','SPI','HSI',
    *[f'Rain_lag_{w}' for w in [1,3,7,30]],
    *[f'Temp_lag_{w}' for w in [1,3,7,30]],
    'Dist_to_Capital','Pop_Density',
    *[f'{c}_norm' for c in vuln_cols]
]
# Impute remaining NaNs
climate_df[scale_cols] = SimpleImputer(strategy='mean').fit_transform(climate_df[scale_cols])
scaler_std = StandardScaler()
climate_df[scale_cols] = scaler_std.fit_transform(climate_df[scale_cols])

# 8. Dimensionality Reduction
ipca = IncrementalPCA(n_components=5, batch_size=5000)
pc = ipca.fit_transform(climate_df[scale_cols])
pc_df = pd.DataFrame(pc, columns=[f'PC{i+1}' for i in range(5)], index=climate_df.index)
climate_df = pd.concat([climate_df, pc_df], axis=1)
print('IncrementalPCA variance explained:', ipca.explained_variance_ratio_)

# 9. Climate Zone Assignment (Vectorized)
elev_df = pd.read_csv(ELEV_PATH)
elev_df['District'] = elev_df['DISTRICT'].str.title().astype('category')
climate_df = climate_df.merge(
    elev_df[['District','Synthetic_Elevation']], on='District', how='left'
)
conds = [
    (climate_df['Synthetic_Elevation']<500)&(climate_df['Temperature (°C)']>25)&(climate_df['Precipitation (mm)']>1500),
    (climate_df['Synthetic_Elevation']<500),
    (climate_df['Synthetic_Elevation'].between(500,2000))&(climate_df['Temperature (°C)']>15)&(climate_df['Precipitation (mm)']>1000),
    (climate_df['Synthetic_Elevation']<2000)
]
choices = ['Tropical','Subtropical','Temperate','Subtemperate']
climate_df['climate_zone'] = pd.Categorical(
    np.select(conds, choices, default='Alpine'),
    categories=['Tropical','Subtropical','Temperate','Subtemperate','Alpine'], ordered=False
)

# 10. Target Creation
# Flood event: SPI threshold
climate_df['flood_event'] = (climate_df['SPI'] > 1.0).astype('uint8')
# Vulnerability class by tercile on sum of norms
climate_df['vuln_score'] = climate_df[[f'{c}_norm' for c in vuln_cols]].sum(axis=1)
climate_df['vulnerability_class'] = pd.qcut(
    climate_df['vuln_score'], 3, labels=['Low','Medium','High']
)
# Impact assessment target: Synthetic continuous impact score
# Define Impact as weighted sum: Flood_event*SPI absolute + vulnerability score
climate_df['Impact'] = (
    climate_df['flood_event'] * climate_df['SPI'].abs() + climate_df['vuln_score']
)
# Drop intermediate vuln_score
climate_df.drop(columns=['vuln_score'], inplace=True)

# 11. Save
os.makedirs(OUTPUT_PATH, exist_ok=True)
climate_df.to_csv(ENGINEERED_FILE, index=False)


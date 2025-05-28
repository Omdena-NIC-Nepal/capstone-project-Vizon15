import numpy as np
import pandas as pd

# -----------------------------
# 1. District to Province Mapping
# -----------------------------
district_to_province_77 = {
    # Province 1 (Koshi Province)
    'Taplejung': 'Koshi', 'Panchthar': 'Koshi', 'Ilam': 'Koshi', 'Jhapa': 'Koshi',
    'Morang': 'Koshi', 'Sunsari': 'Koshi', 'Dhankuta': 'Koshi', 'Terhathum': 'Koshi',
    'Sankhuwasabha': 'Koshi', 'Bhojpur': 'Koshi', 'Okhaldhunga': 'Koshi',
    'Khotang': 'Koshi', 'Udayapur': 'Koshi', 'Solukhumbu': 'Koshi',

    # Province 2 (Madhesh Province)
    'Saptari': 'Madhesh', 'Siraha': 'Madhesh', 'Dhanusha': 'Madhesh',
    'Mahottari': 'Madhesh', 'Sarlahi': 'Madhesh', 'Rautahat': 'Madhesh',
    'Bara': 'Madhesh', 'Parsa': 'Madhesh',

    # Province 3 (Bagmati Province)
    'Sindhuli': 'Bagmati', 'Ramechhap': 'Bagmati', 'Dolakha': 'Bagmati',
    'Sindhupalchok': 'Bagmati', 'Kavrepalanchok': 'Bagmati', 'Lalitpur': 'Bagmati',
    'Bhaktapur': 'Bagmati', 'Kathmandu': 'Bagmati', 'Nuwakot': 'Bagmati',
    'Rasuwa': 'Bagmati', 'Dhading': 'Bagmati', 'Makwanpur': 'Bagmati',
    'Chitwan': 'Bagmati',

    # Province 4 (Gandaki Province)
    'Gorkha': 'Gandaki', 'Lamjung': 'Gandaki', 'Tanahun': 'Gandaki',
    'Kaski': 'Gandaki', 'Manang': 'Gandaki', 'Mustang': 'Gandaki',
    'Parbat': 'Gandaki', 'Syangja': 'Gandaki', 'Baglung': 'Gandaki',
    'Myagdi': 'Gandaki', 'Nawalparasiwest': 'Gandaki',

    # Province 5 (Lumbini Province)
    'Rukumeast': 'Lumbini', 'Rolpa': 'Lumbini', 'Pyuthan': 'Lumbini',
    'Dang': 'Lumbini', 'Banke': 'Lumbini', 'Bardiya': 'Lumbini',
    'Kapilvastu': 'Lumbini', 'Arghakhanchi': 'Lumbini', 'Gulmi': 'Lumbini',
    'Palpa': 'Lumbini', 'Rupandehi': 'Lumbini', 'Nawalparasieast': 'Lumbini',

    # Province 6 (Karnali Province)
    'Rukumwest': 'Karnali', 'Salyan': 'Karnali', 'Dolpa': 'Karnali',
    'Mugu': 'Karnali', 'Humla': 'Karnali', 'Jumla': 'Karnali', 'Kalikot': 'Karnali',
    'Jajarkot': 'Karnali', 'Dailekh': 'Karnali', 'Surkhet': 'Karnali',

    # Province 7 (Sudurpashchim Province)
    'Bajura': 'Sudurpashchim', 'Achham': 'Sudurpashchim', 'Bajhang': 'Sudurpashchim',
    'Darchula': 'Sudurpashchim', 'Baitadi': 'Sudurpashchim', 'Dadeldhura': 'Sudurpashchim',
    'Doti': 'Sudurpashchim', 'Kailali': 'Sudurpashchim', 'Kanchanpur': 'Sudurpashchim'
}

# Extract the list of districts (keys remain in title case)
districts = list(district_to_province_77.keys())

# -----------------------------
# 2. Create Date Range and Grid with Title Case Columns
# -----------------------------
# Generate a daily date range from 2000-01-01 to 2024-12-31.
dates = pd.date_range(start='2000-01-01', end='2024-12-31', freq='D')

# Create a grid for all Date and District combinations using title-case column names.
grid = pd.MultiIndex.from_product([dates, districts], names=["Date", "District"])
df = pd.DataFrame(index=grid).reset_index()

# Map Province for each District.
df["Province"] = df["District"].map(district_to_province_77)

# Set seed for reproducibility.
np.random.seed(42)

# -----------------------------
# 3. Helper Column for Time Trends
# -----------------------------
total_days = (df["Date"].max() - df["Date"].min()).days
df["Days Since Start"] = (df["Date"] - df["Date"].min()).dt.days
frac_time = df["Days Since Start"] / total_days

# -----------------------------
# 4. Generate Synthetic Data Columns (All columns use Title Case names)
# -----------------------------

# --- Environmental Data ---

## 4.1 LULC Index (Note: Acronyms can be kept uppercase if desired)
# Here we simulate the Land Use/Land Cover index as increasing over time with noise.
df["LULC Index"] = 100 + 20 * frac_time + np.random.normal(0, 1, len(df))

## 4.2 River Discharge (m³/s)
# Uses a sine function to mimic monsoon seasonality with a district-specific offset.
district_offset = {district: np.random.uniform(-10, 10) for district in districts}
df["River Discharge"] = (
    50
    + 30 * np.sin(2 * np.pi * df["Date"].dt.dayofyear / 365)
    + df["District"].map(district_offset)
    + np.random.normal(0, 5, len(df))
)

## 4.3 Glacial Lake Area (km²)
# Simulates a slow growing trend with noise and clips any negative values.
df["Glacial Lake Area"] = (
    1 + 0.005 * df["Days Since Start"] + np.random.normal(0, 0.05, len(df))
).clip(lower=0)

## 4.4 Forest Cover (%) and Deforestation Rate (% loss per day)
# Forest Cover declines gradually over time and the deforestation rate adds daily noise.
df["Forest Cover"] = 70 - 5 * frac_time + np.random.normal(0, 1, len(df))
df["Deforestation Rate"] = np.random.normal(0.02, 0.01, len(df)).clip(min=0)

# --- Socioeconomic Data ---

## 4.5 Agri Yield (tons per hectare)
# Incorporates seasonality to simulate crop cycles along with random variations.
df["Agri Yield"] = 2.5 + 0.5 * np.sin(2 * np.pi * df["Date"].dt.dayofyear / 365) + np.random.normal(0, 0.2, len(df))

## 4.6 Pop Vulnerable (Population in Climate-Vulnerable Areas)
# Each district is assigned a constant vulnerable population selected randomly within a plausible range.
district_population = {district: np.random.randint(5000, 50000) for district in districts}
df["Pop Vulnerable"] = df["District"].map(district_population)

## 4.7 Infra Flood (Infrastructure Events in Flood-Prone Regions)
# Simulates sporadic flood events during the monsoon season (approximately day 150-250).
random_vals = np.random.rand(len(df))
is_monsoon = (df["Date"].dt.dayofyear > 150) & (df["Date"].dt.dayofyear < 250)
df["Infra Flood"] = np.where(
    is_monsoon & (random_vals > 0.95),
    np.random.randint(1, 5, size=len(df)),
    0
)

## 4.8 Econ Impact (Economic Impact from Climate Disasters in USD)
# Assigns a random economic loss on days when flood events occur.
df["Econ Impact"] = df["Infra Flood"].apply(lambda x: np.random.randint(10000, 100000) if x > 0 else 0)

# -----------------------------
# 5. Cleanup & Export
# -----------------------------
# Remove the helper column if it is not needed in the final dataset.
df.drop(columns=["Days Since Start"], inplace=True)

# Optionally export to CSV (Caution: This dataset is very large)
df.to_csv('synthetic_nepal_daily_data.csv', index=False)

# Display the first few rows as a verification check.
print(df.head())

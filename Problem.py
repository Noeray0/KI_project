# Import necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Ensure inline plotting


# Load datasets
price_df = pd.read_csv('archive/price.csv')
features_df = pd.read_csv('archive/features.csv')
depreciation_df = pd.read_csv('archive/depreciation.csv')
applications_df = pd.read_csv('archive/applications.csv')
models_df = pd.read_csv('archive/models.csv')
fuel_df = pd.read_csv('archive/fuel.csv')
comfort_features_df = pd.read_csv('archive/comfort_features.csv')
extra_options_df = pd.read_csv('archive/extra_options.csv')
agreement_df = pd.read_csv('archive/agreement.csv')
colors_df = pd.read_csv('archive/colors.csv')
primary_features_df = pd.read_csv('archive/primary_features.csv')
gear_df = pd.read_csv('archive/gear.csv')
locations_df = pd.read_csv('archive/locations.csv')
mans_df = pd.read_csv('archive/mans.csv')
print("done")
# Check for missing values in the price dataset

# Check for missing values in the price dataset
price_df.isnull().sum()

# Convert date columns to datetime in applications_df
applications_df['upload_date'] = pd.to_datetime(applications_df['upload_date'], errors='coerce')
applications_df['insert_date'] = pd.to_datetime(applications_df['insert_date'], errors='coerce')

# Visualize the distribution of car prices
sns.histplot(price_df['price'], bins=50, kde=True)
plt.title('Distribution of Car Prices')
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.show()
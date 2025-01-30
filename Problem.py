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
price_df.isnull().sum()

# Convert date columns to datetime in applications_df
applications_df['upload_date'] = pd.to_datetime(applications_df['upload_date'], errors='coerce')
applications_df['insert_date'] = pd.to_datetime(applications_df['insert_date'], errors='coerce')

# Merge datasets for model building
merged_df = pd.merge(price_df, depreciation_df, on='app_id')

# Select features and target variable
features = ['car_run_km', 'engine_volume', 'cylinders', 'airbags']
X = merged_df[features]
y = merged_df['price']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"mean squared error (whatever that is):{mse}, r2_score:{r2}")

print("done")
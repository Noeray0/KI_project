# https://www.kaggle.com/datasets/qubdidata/auto-market-dataset

# Import necessary libraries
import pandas as pd
import numpy as np

from keras.models import Sequential
from keras.layers import Input, Dense, Dropout
from sklearn.model_selection import train_test_split
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

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

print(y_train.shape)

# Initialize and train the model
model = Sequential()
model.add(Input(shape = (X_train.shape[1],)))
model.add(Dense(64, activation = 'relu'))
model.add(Dropout(0.1))
model.add(Dense(32, activation = 'relu'))
model.add(Dense(1))


model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)

loss, accuracy = model.evaluate(X_test, y_test)
print(f'Loss: {loss}, Accuracy: {accuracy}')

print("done")
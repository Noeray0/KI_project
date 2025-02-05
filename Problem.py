# https://www.kaggle.com/datasets/qubdidata/auto-market-dataset

# Import necessary libraries
import pandas as pd
import numpy as np

from keras.models import Sequential
from keras.layers import Input, Dense, Dropout
from keras.callbacks import EarlyStopping
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
features = ['car_run_km', 'engine_volume', 'prod_year', 'cylinders', 'airbags']
X = merged_df[features]
y = merged_df['price']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = Sequential()
model.add(Input(shape = (X_train.shape[1],)))
model.add(Dense(64, activation = 'relu'))
model.add(Dropout(0.1))
model.add(Dense(32, activation = 'relu'))
model.add(Dense(1))

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
#train the model
model.fit(X_train, y_train, epochs=60, batch_size=32, validation_split=0.2, callbacks=[early_stopping])

loss, accuracy = model.evaluate(X_test, y_test)
print(f'Loss: {loss}, Accuracy: {accuracy}')

print("Model training done")

# Now, ask the user for the features they want to input
print("\nPlease answer the following questions to get a price estimate:")

Continue = 'y'

while Continue == 'y':
    # Create an empty dictionary to store the user's inputs
    user_input = {}

    # Ask for user input for each feature
    for feature in features:
        if feature == 'car_run_km':
            user_input[feature] = int(input("Enter the car's mileage (in km): "))
        elif feature == 'engine_volume':
            user_input[feature] = float(input("Enter the car's engine volume (in liters): "))
        elif feature == 'prod_year':
            user_input[feature] = int(input("Enter the year of production: "))
        elif feature == 'cylinders':
            user_input[feature] = int(input("Enter the number of cylinders in the engine: "))
        elif feature == 'airbags':
            user_input[feature] = int(input("Enter the number of airbags in the car: "))

    # Convert user input into a format suitable for prediction
    input_data = np.array([[user_input['car_run_km'], user_input['engine_volume'], user_input['prod_year'], user_input['cylinders'], user_input['airbags']]])

    # Use the trained model to predict the price
    predicted_price = model.predict(input_data)

    print(f"\nThe estimated price of the car based on the given features is: ${predicted_price[0][0]:.2f}")

    Continue = input("Do you wish to continue (y/n):")
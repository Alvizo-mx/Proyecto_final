import kagglehub
import streamlit as st 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

st.markdown(""" <h1 style='text-align: center;'>Price Flight Predictor</h1> """, unsafe_allow_html=True)
st.write('Jessica Miramontes & Daniel Alvizo')

@st.cache_data
def load_data():
    path = kagglehub.dataset_download("shubhambathwal/flight-price-prediction")
    df_raw = pd.read_csv('/home/codespace/.cache/kagglehub/datasets/shubhambathwal/flight-price-prediction/versions/2/Clean_Dataset.csv')
    df_interim = (
        df_raw.copy()
        .set_axis(
            df_raw.columns.str.replace(' ', '_')
            .str.replace(r'\W', '', regex=True) 
            .str.lower() 
            .str.slice(0, 40), axis=1 
        )
        .rename(columns={'price': 'target'})
        .iloc[:, 1:]
        .drop("flight", axis=1) 
        .astype({
            "airline": "category", 
            "source_city": "category", 
            "departure_time": "category", 
            "stops": "category", 
            "arrival_time": "category", 
            "destination_city": "category", 
            "class": "category"
        })
    )
    df = (
        df_interim.copy()
        .reindex(
            columns=(
                ['target'] + 
                [c for c in df_interim.columns.to_list() if c != 'target']
            )
        )
    )
    return df

df_ch = load_data()

def model_creation():
    inputs_cols = ['airline', 'source_city', 'departure_time', 'stops', 'arrival_time',
                   'destination_city', 'class', 'duration', 'days_left']
    targets_col = 'target'
    inputs_dataset = df_ch[inputs_cols].copy()
    targets_set = df_ch[targets_col].copy()
    numeric_cols = inputs_dataset.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = inputs_dataset.select_dtypes(include='category').columns.tolist()
    scaler = MinMaxScaler()
    scaler.fit(inputs_dataset[numeric_cols])
    inputs_dataset[numeric_cols] = scaler.transform(inputs_dataset[numeric_cols])
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoder.fit(inputs_dataset[categorical_cols])
    encoder_cols = encoder.get_feature_names_out(categorical_cols)
    inputs_dataset[encoder_cols] = encoder.transform(inputs_dataset[categorical_cols])
    X = pd.concat([inputs_dataset[numeric_cols], inputs_dataset[encoder_cols]], axis=1)
    y = targets_set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    return X_train, X_test, y_train, y_test, encoder, scaler

X_train, X_test, y_train, y_test, encoder, scaler = model_creation()

@st.cache_resource
def train_model(X_train, y_train):
    rfreg = RandomForestRegressor()
    rfreg.fit(X_train, y_train)
    model_dict = {
        'model': rfreg,
        'predictors': X_train.columns.to_list(),
        'target_name': y_train.name,
        'algorithm_name': 'Random Forest Regression'
    }
    return model_dict

# Create the Random Forest model dictionary
rf_model_dict = train_model(X_train, y_train)

# Use the trained model to make predictions
def predict_with_model(model_dict, X_test):
    model = model_dict['model']
    predictions = model.predict(X_test)
    return predictions

# Make predictions
pred_rf = predict_with_model(rf_model_dict, X_test)

st.title("Flight Price Predictor")

st.write("Enter values for the following features to predict the flight price:")

# User input for features
airline = st.selectbox('Airline', df_ch['airline'].unique())
source_city = st.selectbox('Source City', df_ch['source_city'].unique())
departure_time = st.selectbox('Departure Time', df_ch['departure_time'].unique())
stops = st.selectbox('Stops', df_ch['stops'].unique())
arrival_time = st.selectbox('Arrival Time', df_ch['arrival_time'].unique())
destination_city = st.selectbox('Destination City', df_ch['destination_city'].unique())
seat_class = st.selectbox('Class', df_ch['class'].unique())
duration = st.number_input('Duration (in hours)', min_value=0.0, max_value=50.0, step=0.1)
days_left = st.number_input('Days Left to Departure', min_value=0, max_value=365, step=1)

input_data = pd.DataFrame({
    'airline': [airline],
    'source_city': [source_city],
    'departure_time': [departure_time],
    'stops': [stops],
    'arrival_time': [arrival_time],
    'destination_city': [destination_city],
    'class': [seat_class],
    'duration': [duration],
    'days_left': [days_left]
})

# Preprocess input data
numeric_cols = ['duration', 'days_left']
categorical_cols = ['airline', 'source_city', 'departure_time', 'stops', 'arrival_time', 'destination_city', 'class']

# Transform input data
input_data[numeric_cols] = scaler.transform(input_data[numeric_cols])
input_data_encoded = pd.DataFrame(encoder.transform(input_data[categorical_cols]), columns=encoder.get_feature_names_out(categorical_cols))
# Combine numeric and encoded categorical features
X_input = pd.concat([input_data[numeric_cols], input_data_encoded], axis=1)
# Load trained model from model_dict for prediction
model = rf_model_dict['model']
prediction = model.predict(X_input)

st.write(f"The predicted flight price is: ${prediction[0]:.2f}")

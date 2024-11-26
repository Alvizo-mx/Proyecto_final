# Step 0. Import libraries, custom modules and logging
import kagglehub
import streamlit as st 
# Data -----------------------------------------------------------------
import pandas as pd
import numpy as np
# Graphics -------------------------------------------------------------
import matplotlib.pyplot as plt
import seaborn as sns
# Machine learning -----------------------------------------------------
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (mean_squared_error,
                             mean_absolute_error,
                             r2_score,
                             root_mean_squared_error)
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import (OneHotEncoder,
                                   MinMaxScaler,
                                  )


st.markdown(""" <h1 style='text-align: center;'>Price Flight Predictions</h1> """, unsafe_allow_html=True)
st.markdown(""" <h2 style='text-align: center;'>Using Machine learning </h2> """, unsafe_allow_html=True)

st.write('Jessica Miramontes & Daniel Alvizo')
st.write("""
The purpose of this study is to analyze the flight booking dataset from the “Ease My Trip” website,
using various statistical hypothesis tests to see which variables affect the most. Then, machine
learning algorithms will predict the prices and compare them to see which is more effective for this task.    
        
         """)



def load_data():
    # Assuming kagglehub is configured and installed correctly
    path = kagglehub.dataset_download("shubhambathwal/flight-price-prediction")
    
    # Load the dataset
    df_raw = pd.read_csv('/home/codespace/.cache/kagglehub/datasets/shubhambathwal/flight-price-prediction/versions/2/Clean_Dataset.csv')

    # Clean and preprocess the dataset
    df_interim = (
        df_raw
        .copy()
        .set_axis(
            df_raw.columns.str.replace(' ', '_') # Replace spaces with _
            .str.replace(r'\W', '', regex=True) # Remove non-alphanumeric characters
            .str.lower() # Convert to lowercase
            .str.slice(0, 40), axis=1 # Limit column names to 40 characters
        )
        .rename(columns={'price': 'target'}) # Rename 'price' to 'target'
        .iloc[:, 1:] # Exclude the first column
        .drop("flight", axis=1) # Drop the 'flight' column
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
    
    # Reindex columns to place 'target' first
    df = (
        df_interim
        .copy()
        .reindex(
            columns=(
                ['target'] + 
                [c for c in df_interim.columns.to_list() if c != 'target']
            )
        )
    )
    
    return df



df_ch = load_data()
st.dataframe(df_ch.sample(5))
st.write(f"""
   As we can see, this is a sample of the data frama we use 
   for our flight price prediction, and with statistical analysis
   We will see what whats going on
         
         """)

def load_analysis():
    df_train, df_test = train_test_split(df_ch,
                                     random_state=2024,
                                     test_size=0.2)
    df_train = df_train.reset_index(drop=True).sort_values(by='target')
    return df_train
    
df_train = load_analysis()
st.write(df_train.describe(include='category').T) 
st.write(df_train.describe().T)

st.write(f'Now we will perform EDA With both categorical and numerical Variables')

fig, ax = plt.subplots()
df_train.hist(ax=ax)
st.pyplot(fig)

st.write(f'As we can see, this are the histograms for numerical variables to analyze')

fig, axis = plt.subplots(3, 2, figsize=(14, 12))
sns.histplot(ax=axis[0, 0], data=df_train, x="airline")
sns.histplot(ax=axis[0, 1], data=df_train, x="source_city")
sns.histplot(ax=axis[1, 0], data=df_train, x="departure_time")
sns.histplot(ax=axis[1, 1], data=df_train, x="stops")
sns.histplot(ax=axis[2, 0], data=df_train, x="arrival_time")
sns.histplot(ax=axis[2, 1], data=df_train, x="destination_city")
st.pyplot(fig)

fig, axis = plt.subplots(3, 2, figsize = (10, 7))
sns.histplot(ax = axis[0, 0], data = df_train, x= "target").set(xlabel = None)
sns.boxplot(ax = axis[0, 1], data = df_train, x = "target")
sns.histplot(ax = axis[1, 0], data = df_train, x = "duration").set(xlabel = None, ylabel = None)
sns.boxplot(ax = axis[1, 1], data = df_train, x = "duration")
sns.histplot(ax = axis[2, 0], data = df_train, x = "days_left").set(xlabel = None, ylabel = None)
st.pyplot(fig)

fig, axis = plt.subplots(2, 2, figsize=(10, 8))
sns.regplot(ax=axis[0, 0], data=df_train, x="target", y="duration")
sns.heatmap(df_train[["target", "duration"]].corr(), annot=True, fmt=".2f", ax=axis[1, 0], cbar=False)
sns.regplot(ax=axis[0, 1], data=df_train, x="target", y="days_left").set(ylabel=None)
sns.heatmap(df_train[["target", "days_left"]].corr(), annot=True, fmt=".2f", ax=axis[1, 1])
st.pyplot(fig)

fig, axis = plt.subplots(2, 3, figsize = (15, 7))
sns.countplot(ax = axis[0, 0], data = df_train, x = "airline", hue = "class")
sns.countplot(ax = axis[0, 1], data = df_train, x = "source_city", hue = "class").set(ylabel = None)
sns.countplot(ax = axis[0, 2], data = df_train, x = "destination_city", hue = "class").set(ylabel = None)
sns.countplot(ax = axis[1, 0], data = df_train, x = "departure_time", hue = "class")
sns.countplot(ax = axis[1, 1], data = df_train, x = "stops", hue = "class").set(ylabel = None)
sns.countplot(ax = axis[1, 2], data = df_train, x = "arrival_time", hue = "class").set(ylabel = None)
st.pyplot(fig)

def model_creation():
    inputs_cols=['airline',  'source_city', 'departure_time','stops', 'arrival_time',
            'destination_city', 'class', 'duration', 'days_left']
    targets_col='target'
    inputs_dataset = df_ch[inputs_cols].copy()
    targets_set    = df_ch[targets_col].copy()
    numeric_cols = inputs_dataset.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = inputs_dataset.select_dtypes(include='category').columns.tolist()
    scaler = MinMaxScaler()
    scaler.fit(inputs_dataset[numeric_cols])
    inputs_dataset[numeric_cols] = scaler.transform(inputs_dataset[numeric_cols])
    encoder = OneHotEncoder(sparse_output=False,handle_unknown='ignore')
    encoder.fit(inputs_dataset[categorical_cols])
    encoder_cols = encoder.get_feature_names_out(categorical_cols)
    inputs_dataset[encoder_cols]=encoder.transform(inputs_dataset[categorical_cols])
    X = pd.concat([inputs_dataset[numeric_cols], inputs_dataset[encoder_cols]],axis=1)
    y = targets_set
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.20,random_state=42)
    lr = LinearRegression()
    lr.fit(X_train,y_train)
    pred = lr.predict(X_test)
    lr_score= r2_score(y_test,pred)
    return lr_score, X_train, X_test, y_train, y_test, pred

lr_score, X_train, X_test, y_train, y_test, pred = model_creation()

st.write(f'Linear regression accuracy score: {lr_score}')
fig, ax = plt.subplots()
ax.scatter(x= y_test, y= pred, c= 'k') 
ax.plot([0, 40000], [-3000, 20000], c= 'r')
ax.plot([10000, 120000], [45000, 60000], c= 'r')
ax.axis('equal')
ax.set_xlabel('Real')
ax.set_ylabel('Predicted')
st.pyplot(fig)

def decision_tree_regressor():
    dtr = DecisionTreeRegressor()
    dtr.fit(X_train, y_train)
    pred_dtr = dtr.predict(X_test)
    dtr_score = r2_score(y_test, pred_dtr)
    return dtr_score, y_test, pred_dtr

dtr_score, y_test, pred_dtr = decision_tree_regressor()
st.write(f'Decision Tree Regressor accuracy score: {dtr_score}')

# Mostrar gráfico de dispersión para Decision Tree Regressor
fig, ax = plt.subplots()
ax.scatter(x=y_test, y=pred_dtr, c='k')
ax.plot([0, 140000], [0, 120000], c='r')
ax.axis('equal')
ax.set_xlabel('Real')
ax.set_ylabel('Predicted')
st.pyplot(fig)

def random_forest_regressor():
    rf = RandomForestRegressor()
    rf.fit(X_train, y_train)
    pred_rf = rf.predict(X_test) 
    rf_score = r2_score(y_test, pred_rf)
    return rf_score, pred_rf 

rf_score, pred_rf = random_forest_regressor() 
st.write(f'Random Forest Regressor accuracy score: {rf_score}') 
# Mostrar gráfico de dispersión para Random Forest Regressor 
fig, ax = plt.subplots() 
ax.scatter(x=y_test, y=pred_rf, c='k')
ax.plot([0, 140000], [0, 120000], c='r')
ax.axis('equal')
ax.set_xlabel('Real') 
ax.set_ylabel('Predicted') 
st.pyplot(fig)




    
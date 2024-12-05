# Step 0. Import libraries, custom modules and logging
import kagglehub
import streamlit as st 
from PIL import Image
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
st.markdown("""
    <div style="text-align: center;">
        The purpose of this study is to analyze the flight booking dataset from the “Ease My Trip” website,
        using various statistical hypothesis tests to see which variables affect the most. Then, machine
        learning algorithms will predict the prices and compare them to see which is more effective for this task.
    </div>
    """, unsafe_allow_html=True)



def load_data():
    path = kagglehub.dataset_download("shubhambathwal/flight-price-prediction")
    df_raw = pd.read_csv('/home/codespace/.cache/kagglehub/datasets/shubhambathwal/flight-price-prediction/versions/2/Clean_Dataset.csv')

    df_interim = (
        df_raw
        .copy()
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
st.header('Price flight Dataframe')
st.dataframe(df_ch.sample(5))
st.write(f"""
   As we can see, this is a sample of the data frama we use 
   for our flight price prediction, and with statistical analysis
   We will see what whats going on
         
         """)

st.write("""
## **Data Dictionary**
|Name|Description|Units|Type|
|----|-----------|-----|----|
|airline|The name of the airline company is stored in the airline column. It is a categorical feature having 6 different airlines.|none|category|
|flight|Flight stores information regarding the plane's flight code. It is a categorical feature.|none|category|
|Source City|City from which the flight takes off. It is a categorical feature having 6 unique cities.|none|category|
|Departure Time|This is a derived categorical feature obtained created by grouping time periods into bins. It stores information about the departure time and have 6 unique time labels.|none|category|
|Stops| A categorical feature with 3 distinct values that stores the number of stops between the source and destination cities.|none|category|
|Arrival Time| This is a derived categorical feature created by grouping time intervals into bins. It has six distinct time labels and keeps information about the arrival time.|none|category|
|Destination City| City where the flight will land. It is a categorical feature having 6 unique cities.|none|category|
|Class| A categorical feature that contains information on seat class; it has two distinct values: Business and Economy.|none|category|
|Duration| A continuous feature that displays the overall amount of time it takes to travel between cities in hours.|none|float|
|Price| Target variable stores information of the ticket price.|none|int|


         """)


def load_analysis():
    df_train, df_test = train_test_split(df_ch,
                                     random_state=2024,
                                     test_size=0.2)
    df_train = df_train.reset_index(drop=True).sort_values(by='target')
    return df_train
    
df_train = load_analysis()
st.header(f'categorical analysis')
st.write(df_train.describe(include='category').T) 
st.header(f'numerical analysis')
st.write(df_train.describe().T)
st.header(f'Exploratory Data Analysis ')
st.write(f'Now we will perform EDA With both categorical and numerical Variables and see what we can see from this Dataframe')
st.title(f'numerical')

fig, ax = plt.subplots()
df_train.hist(ax=ax)
st.pyplot(fig)

fig, axis = plt.subplots(3, 2, figsize = (10, 7))
sns.histplot(ax = axis[0, 0], data = df_train, x= "target").set(xlabel = None)
sns.boxplot(ax = axis[0, 1], data = df_train, x = "target")
sns.histplot(ax = axis[1, 0], data = df_train, x = "duration").set(xlabel = None, ylabel = None)
sns.boxplot(ax = axis[1, 1], data = df_train, x = "duration")
sns.histplot(ax = axis[2, 0], data = df_train, x = "days_left").set(xlabel = None, ylabel = None)
sns.boxplot(ax = axis[2, 1], data = df_train, x = "days_left")
st.pyplot(fig)


st.title(f'categorical')

fig, axis = plt.subplots(3, 2, figsize=(14, 12))
sns.histplot(ax=axis[0, 0], data=df_train, x="airline")
sns.histplot(ax=axis[0, 1], data=df_train, x="source_city")
sns.histplot(ax=axis[1, 0], data=df_train, x="departure_time")
sns.histplot(ax=axis[1, 1], data=df_train, x="stops")
sns.histplot(ax=axis[2, 0], data=df_train, x="arrival_time")
sns.histplot(ax=axis[2, 1], data=df_train, x="destination_city")
st.pyplot(fig)

st.title(f'Bivariate analysis of numerical variables')
fig, axis = plt.subplots(2, 2, figsize=(10, 8))
sns.regplot(ax=axis[0, 0], data=df_train, x="target", y="duration")
sns.heatmap(df_train[["target", "duration"]].corr(), annot=True, fmt=".2f", ax=axis[1, 0], cbar=False)
sns.regplot(ax=axis[0, 1], data=df_train, x="target", y="days_left").set(ylabel=None)
sns.heatmap(df_train[["target", "days_left"]].corr(), annot=True, fmt=".2f", ax=axis[1, 1])
st.pyplot(fig)

st.title(f'Bivariate analysis of categorical variables')
fig, axis = plt.subplots(2, 3, figsize = (15, 7))
sns.countplot(ax = axis[0, 0], data = df_train, x = "airline", hue = "class")
sns.countplot(ax = axis[0, 1], data = df_train, x = "source_city", hue = "class").set(ylabel = None)
sns.countplot(ax = axis[0, 2], data = df_train, x = "destination_city", hue = "class").set(ylabel = None)
sns.countplot(ax = axis[1, 0], data = df_train, x = "departure_time", hue = "class")
sns.countplot(ax = axis[1, 1], data = df_train, x = "stops", hue = "class").set(ylabel = None)
sns.countplot(ax = axis[1, 2], data = df_train, x = "arrival_time", hue = "class").set(ylabel = None)
st.pyplot(fig)

st.title(f'Bivariate analysis of categorical & numerical variables')
fig = sns.pairplot(data=df_train, hue='class')
st.pyplot(fig)

st.title(f'machine learning model')
st.title(f'Linear regression')


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
    return X_train, X_test, y_train, y_test, numeric_cols,categorical_cols,encoder_cols

X_train, X_test, y_train, y_test, numeric_cols,categorical_cols,encoder_cols = model_creation()

def linear_regression():
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    pred = lr.predict(X_test)
    lr_score = r2_score(y_test, pred)
    lr_mse = mean_squared_error(y_test, pred)
    lr_rmse = np.sqrt(lr_mse)
    lr_mae = mean_absolute_error(y_test, pred)
    return lr_score, lr_mse, lr_rmse, lr_mae, pred

lr_score, lr_mse, lr_rmse, lr_mae, pred = linear_regression()
st.write(f"""
Linear regression is a type of supervised machine learning algorithm that computes the linear relationship
between the dependent variable and one or more independent features by fitting a linear equation to
observed data.
""")

st.write(f'Linear regression accuracy score: {lr_score * 100:.2f}%')
fig, ax = plt.subplots()
ax.scatter(x=y_test, y=pred, c='k')
ax.plot([0, 40000], [-3000, 20000], c='r')
ax.plot([10000, 120000], [45000, 60000], c='r')
ax.axis('equal')
ax.set_xlabel('Real')
ax.set_ylabel('Predicted')
st.pyplot(fig)

st.title('Decision Tree Regressor')
st.write(f"""
Decision Tree is a decision-making tool that uses a flowchart-like tree structure or is a model of
decisions and all of their possible results, including outcomes, input costs, and utility.
""")

def decision_tree_regressor():
    dtr = DecisionTreeRegressor()
    dtr.fit(X_train, y_train)
    pred_dtr = dtr.predict(X_test)
    dtr_score = r2_score(y_test, pred_dtr)
    dtr_mse = mean_squared_error(y_test, pred_dtr)
    dtr_rmse = np.sqrt(dtr_mse)
    dtr_mae = mean_absolute_error(y_test, pred_dtr)
    return dtr_score, dtr_mse, dtr_rmse, dtr_mae, pred_dtr

dtr_score, dtr_mse, dtr_rmse, dtr_mae, pred_dtr = decision_tree_regressor()
st.write(f'Decision Tree Regressor accuracy score: {dtr_score * 100:.2f}%')

fig, ax = plt.subplots()
ax.scatter(x=y_test, y=pred_dtr, c='k')
ax.plot([0, 140000], [0, 120000], c='r')
ax.axis('equal')
ax.set_xlabel('Real')
ax.set_ylabel('Predicted')
st.pyplot(fig)

st.title('Random Forest Regressor')
st.write(f"""
Random Forest Regression is a versatile machine-learning technique for predicting numerical values.
It combines the predictions of multiple decision trees to reduce overfitting and improve accuracy.
Python’s machine-learning libraries make it easy to implement and optimize this approach.
""")

def random_forest_regressor():
    rf = RandomForestRegressor()
    rf.fit(X_train, y_train)
    pred_rf = rf.predict(X_test)
    rf_score = r2_score(y_test, pred_rf)
    rf_mse = mean_squared_error(y_test, pred_rf)
    rf_rmse = np.sqrt(rf_mse)
    rf_mae = mean_absolute_error(y_test, pred_rf)
    return rf_score, rf_mse, rf_rmse, rf_mae, pred_rf

rf_score, rf_mse, rf_rmse, rf_mae, pred_rf = random_forest_regressor()
st.write(f'Random Forest Regressor accuracy score: {rf_score * 100:.2f}%')

fig, ax = plt.subplots()
ax.scatter(x=y_test, y=pred_rf, c='k')
ax.plot([0, 140000], [0, 120000], c='r')
ax.axis('equal')
ax.set_xlabel('Real')
ax.set_ylabel('Predicted')
st.pyplot(fig)

def table_results():
    algorithm_names = ["Linear Regression", "Decision Tree", "Random Forest"]
    mse_values = [lr_mse, dtr_mse, rf_mse]
    rmse_values = [lr_rmse, dtr_rmse, rf_rmse]
    mae_values = [lr_mae, dtr_mae, rf_mae]
    r2_values = [lr_score, dtr_score, rf_score]
    comparison = pd.DataFrame({
        "algorithm": algorithm_names,
        "MSE": mse_values,
        "RMSE": rmse_values,
        "MAE": mae_values,
        "R2": r2_values
    })
    comparison = comparison.round(2)
    return comparison

table_result = table_results()
st.dataframe(table_result)
image = Image.open('permutation_importance.jpg.png') 
st.image(image, caption='permutation importance', use_container_width=True)
st.write(f"""
For this dataset, the model that best predicts the prices of the flights is the Random
Forest Regression which has an R2 score of 0.98.
The second best is the Decision Tree Regression with an R2 score of 0.97.
Finally, Linear Regression with an R2 score of 0.91
We found that there is no single variable that can be used as a predictor for the price and that
actually, it is a combination of variables that make the prediction possible.  
         """)


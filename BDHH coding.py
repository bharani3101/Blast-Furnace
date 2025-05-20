import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import numpy as np
data = pd.read_csv(r"C:\Users\Bharanidharan\OneDrive\Desktop\Final year project\bf3_data_2022_01_07.csv")
data.drop(columns=['SKIN_TEMP_AVG'],inplace=True)
data['DATE_TIME'] = pd.to_datetime(data['DATE_TIME'], format="%d-%m-%y %H:%M")
data.dropna(inplace=True)
data
def train_and_predict(X, y, shift_hours):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X)
    predicted_df = pd.DataFrame(predictions, columns=X.columns)
    predicted_df['DATE_TIME'] = X.index + pd.to_timedelta(shift_hours, unit='h')
    mse = mean_squared_error(y_test, model.predict(X_test))
    mae = mean_absolute_error(y_test, model.predict(X_test))
    r2 = r2_score(y_test, model.predict(X_test))
    print(f"Shift {shift_hours} hours - Mean Squared Error: {mse}")
    print(f"Shift {shift_hours} hours - Mean Absolute Error: {mae}")
    print(f"Shift {shift_hours} hours - R-squared: {r2}")
    return model, predicted_df
X = data.drop(columns=['DATE_TIME'])
X.index = data['DATE_TIME']
y = X.copy()
model_shift_1, predicted_df_shift_1 = train_and_predict(X, y, shift_hours=1)
predicted_df_shift_1['CO_CO2_ratio'] = predicted_df_shift_1['CO'] / predicted_df_shift_1['CO2']
#predicted_df_shift_1.to_csv('predicted_data_after_1_hour.csv', index=False)
predicted_df_shift_1
X_shift_2 = predicted_df_shift_1.drop(columns=['DATE_TIME'])
X_shift_2.index = predicted_df_shift_1['DATE_TIME']
model_shift_2, predicted_df_shift_2 = train_and_predict(X_shift_2, X_shift_2, shift_hours=2)
predicted_df_shift_2['CO_CO2_ratio'] = predicted_df_shift_2['CO'] / predicted_df_shift_2['CO2']
#predicted_df_shift_2.to_csv('predicted_data_after_2_hours.csv', index=False)
predicted_df_shift_2
X_shift_3 = predicted_df_shift_2.drop(columns=['DATE_TIME'])
X_shift_3.index = predicted_df_shift_2['DATE_TIME']
model_shift_3, predicted_df_shift_3 = train_and_predict(X_shift_3, X_shift_3, shift_hours=3)
predicted_df_shift_3['CO_CO2_ratio'] = predicted_df_shift_3['CO'] / predicted_df_shift_3['CO2']
#predicted_df_shift_3.to_csv('predicted_data_after_3_hours.csv', index=False)
predicted_df_shift_3
X_shift_4 = predicted_df_shift_3.drop(columns=['DATE_TIME'])
X_shift_4.index = predicted_df_shift_3['DATE_TIME']
model_shift_4, predicted_df_shift_4 = train_and_predict(X_shift_4, X_shift_4, shift_hours=4)
predicted_df_shift_4['CO_CO2_ratio'] = predicted_df_shift_4['CO'] / predicted_df_shift_4['CO2']
#predicted_df_shift_4.to_csv('predicted_data_after_4_hours.csv', index=False)
predicted_df_shift_4

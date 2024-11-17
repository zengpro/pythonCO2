import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Загрузка сохранённой модели
@st.cache_resource
def load_model(model_path='co2_model.pkl'):
    return joblib.load(model_path)

model = load_model()

# Загрузка данных для выбора стран
@st.cache_data
def load_data(file_path='datasetCO2.csv'):
    data = pd.read_csv(file_path)
    return data

data = load_data()

# Streamlit: интерфейс
st.title("CO2 Emissions Prediction")

st.write("Введите параметры для прогнозирования выбросов CO₂:")

# Ввод параметров
country = st.selectbox("Страна", options=data['Country'].unique())
year = st.number_input("Год", min_value=1960, max_value=2100, value=2025)

if st.button("Прогнозировать"):
    # Прогнозирование
    input_data = pd.DataFrame({'Country': [country], 'Year': [year]})
    prediction = model.predict(input_data)
    st.write(f"Прогноз выбросов CO₂ для {country} в {year}: {prediction[0]:.2f} килотонн")

import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Загрузка модели
@st.cache_resource
def load_model(model_path='co2_model.pkl'):
    return joblib.load(model_path)

model = load_model()

# Загрузка данных
@st.cache_data
def load_data(file_path='datasetCO2.csv'):
    data = pd.read_csv(file_path)
    data['Year'] = pd.to_datetime(data['Date']).dt.year
    return data

data = load_data()

# Streamlit интерфейс
st.title("CO2 Emissions Prediction")

st.write("Введите параметры для прогнозирования выбросов CO₂:")

# Ввод параметров
country = st.selectbox("Страна", options=data['Country'].unique())
year = st.number_input("Год", min_value=int(data['Year'].min()), max_value=2100, value=2025)
metric_tons_per_capita = st.slider(
    "Metric Tons Per Capita",
    min_value=float(data['Metric Tons Per Capita'].min()),
    max_value=float(data['Metric Tons Per Capita'].max()),
    value=float(data['Metric Tons Per Capita'].mean())
)

if st.button("Прогнозировать"):
    # Прогнозирование
    input_data = pd.DataFrame({
        'Country': [country],
        'Year': [year],
        'Metric Tons Per Capita': [metric_tons_per_capita]
    })
    prediction = model.predict(input_data)
    st.write(f"Прогноз выбросов CO₂ для {country} в {year}: {prediction[0]:.2f} килотонн")

    # Визуализация данных для выбранной страны
    st.subheader(f"Динамика выбросов CO₂ для {country}")

    # Фильтрация данных
    country_data = data[data['Country'] == country].sort_values(by='Year')

    # Создание прогнозов для всех лет до выбранного
    past_years = list(range(country_data['Year'].max() + 1, year + 1))
    forecast_input = pd.DataFrame({
        'Country': [country] * len(past_years),
        'Year': past_years,
        'Metric Tons Per Capita': [metric_tons_per_capita] * len(past_years)
    })
    forecast_values = model.predict(forecast_input)

    # Добавление прогнозируемых значений в данные
    forecast_df = pd.DataFrame({'Year': past_years, 'Kilotons of Co2': forecast_values})
    combined_data = pd.concat([country_data[['Year', 'Kilotons of Co2']], forecast_df])

    # График
    st.write("График выбросов CO₂:")
    plt.figure(figsize=(10, 5))
    plt.plot(combined_data['Year'], combined_data['Kilotons of Co2'], marker='o', linestyle='-', label='Actual & Forecasted CO₂')
    plt.axvline(x=year, color='red', linestyle='--', label='Прогнозируемый год')
    plt.title(f"Динамика выбросов CO₂ для {country}")
    plt.xlabel("Год")
    plt.ylabel("Kilotons of CO₂")
    plt.legend()
    plt.grid()
    st.pyplot(plt)

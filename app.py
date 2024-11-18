import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt


# Загрузка сохранённой модели
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
    last_known_year = int(data[data['Country'] == country]['Year'].max())
    future_years = list(range(last_known_year + 1, year + 1))

    future_predictions = []
    for future_year in future_years:
        input_data = pd.DataFrame({
            'Country': [country],
            'Year': [future_year],
            'Metric Tons Per Capita': [metric_tons_per_capita]
        })
        predicted_value = model.predict(input_data)[0]
        future_predictions.append({'Year': future_year, 'Kilotons of Co2': predicted_value})

    # Прогнозируемое значение для выбранного года
    input_data = pd.DataFrame({
        'Country': [country],
        'Year': [year],
        'Metric Tons Per Capita': [metric_tons_per_capita]
    })
    final_prediction = model.predict(input_data)[0]
    st.write(f"Прогноз выбросов CO₂ для {country} в {year}: {final_prediction:.2f} килотонн")

    # Визуализация данных для выбранной страны
    st.subheader(f"Динамика выбросов CO₂ для {country}")

    # Фильтрация данных для страны
    country_data = data[data['Country'] == country].sort_values(by='Year')

    # Добавление прогнозируемых значений
    future_data = pd.DataFrame(future_predictions)
    combined_data = pd.concat([country_data, future_data]).sort_values(by='Year')

    # Соединение последнего известного значения с прогнозом
    last_known_value = country_data.iloc[-1] if not country_data.empty else None
    if last_known_value is not None and not future_data.empty:
        future_data = pd.concat([
            pd.DataFrame([{
                'Year': last_known_value['Year'],
                'Kilotons of Co2': last_known_value['Kilotons of Co2']
            }]),
            future_data
        ]).sort_values(by='Year')

    # График
    st.write("График выбросов CO₂:")
    plt.figure(figsize=(10, 5))
    plt.plot(country_data['Year'], country_data['Kilotons of Co2'], marker='o', label='Исторические данные')
    plt.plot(future_data['Year'], future_data['Kilotons of Co2'], marker='x', linestyle='--', color='orange',
             label='Прогнозируемые данные')
    plt.scatter(year, final_prediction, color='red', label=f'Прогноз на {year}')
    plt.title(f"Динамика выбросов CO₂ для {country}")
    plt.xlabel("Год")
    plt.ylabel("Kilotons of CO₂")
    plt.legend()
    plt.grid()
    st.pyplot(plt)

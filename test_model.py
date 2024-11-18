import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt


# Загрузка модели
def load_model(model_path='co2_model.pkl'):
    return joblib.load(model_path)


model = load_model()


# Проверка модели на прогнозирование значений для одной страны
def test_model_on_country(country, start_year, num_years, metric_tons_per_capita):
    # Создаём тестовый набор данных для прогноза
    test_data = pd.DataFrame({
        'Country': [country] * num_years,
        'Year': [start_year + i for i in range(num_years)],
        'Metric Tons Per Capita': [metric_tons_per_capita] * num_years
    })

    # Прогнозируем
    predictions = model.predict(test_data)

    # Выводим результаты
    results = pd.DataFrame({
        'Year': test_data['Year'],
        'Predicted CO2 (Kilotons)': predictions
    })

    print(results)

    # Визуализация
    plt.figure(figsize=(10, 6))
    plt.plot(results['Year'], results['Predicted CO2 (Kilotons)'], marker='o', linestyle='-', label='Predicted CO₂')
    plt.title(f"Predicted CO₂ Emissions for {country}")
    plt.xlabel("Year")
    plt.ylabel("Kilotons of CO₂")
    plt.grid()
    plt.legend()
    plt.show()


# Параметры для проверки
country_to_test = 'Ecuador'  # Укажите страну для проверки
start_year = 2019  # Год начала прогнозирования
num_years = 10  # Количество лет для прогнозирования
metric_tons_per_capita = 3.03  # Примерное значение Metric Tons Per Capita

test_model_on_country(country_to_test, start_year, num_years, metric_tons_per_capita)

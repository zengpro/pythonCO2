from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from clearml import Task, Logger
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# ClearML: инициализация задачи
task = Task.init(project_name="CO2 Forecasting", task_name="CO2 Model Training")
logger = Logger.current_logger()

# Загрузка данных
def load_data(file_path='datasetCO2.csv'):
    data = pd.read_csv(file_path)
    data['Year'] = pd.to_datetime(data['Date']).dt.year
    return data

data = load_data()

# Признаки и целевая переменная
X = data[['Country', 'Year', 'Metric Tons Per Capita']]
y = data['Kilotons of Co2']

# Создание трансформеров
categorical_features = ['Country']
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

numeric_features = ['Year', 'Metric Tons Per Capita']
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler()),
    ('poly', PolynomialFeatures(degree=2, include_bias=False))  # Полиномиальные признаки
])

# Композиция трансформеров
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', categorical_transformer, categorical_features),
        ('num', numeric_transformer, numeric_features)
    ]
)

# Создание модели
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(random_state=42, n_estimators=200))
])

# Обучение модели
model.fit(X, y)

# Сохранение модели
model_path = 'co2_model.pkl'
joblib.dump(model, model_path)

task.upload_artifact(name="Trained Model", artifact_object=model_path)

# Прогнозирование и оценка модели
y_pred = model.predict(X)
mse = mean_squared_error(y, y_pred)
rmse = mse ** 0.5
mae = mean_absolute_error(y, y_pred)
r2 = r2_score(y, y_pred)

# Логирование метрик
logger.report_scalar("Metrics", "RMSE", iteration=0, value=rmse)
logger.report_scalar("Metrics", "MAE", iteration=0, value=mae)
logger.report_scalar("Metrics", "R2 Score", iteration=0, value=r2)

# Вывод метрик
print(f"RMSE: {rmse:.2f}, MAE: {mae:.2f}, R2 Score: {r2:.2f}")
print("Модель сохранена в 'co2_model.pkl'")

# 1. Тренды выбросов CO₂ по времени
time_trends = data.groupby('Year')['Kilotons of Co2'].mean().reset_index()

# Построение графика трендов выбросов
plt.figure(figsize=(12, 6))
plt.plot(time_trends['Year'], time_trends['Kilotons of Co2'], marker='o', linestyle='-', color='blue')
plt.title("Тренды выбросов CO₂ по времени", fontsize=16)
plt.xlabel("Год", fontsize=14)
plt.ylabel("Средние выбросы CO₂ (Kilotons)", fontsize=14)
plt.grid()
plt.tight_layout()

# Сохранение графика в файл и загрузка в ClearML
plt.savefig("co2_trends.png")
task.upload_artifact("CO2_Trends_Over_Time", artifact_object="co2_trends.png")
plt.show()

# 2. Влияние выбросов на душу населения
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Metric Tons Per Capita', y='Kilotons of Co2', data=data, alpha=0.6)
plt.title("Влияние выбросов на душу населения", fontsize=16)
plt.xlabel("Выбросы на душу населения (Metric Tons Per Capita)", fontsize=14)
plt.ylabel("Общие выбросы CO₂ (Kilotons)", fontsize=14)
plt.grid()
plt.tight_layout()

# Сохранение графика в файл и загрузка в ClearML
plt.savefig("co2_per_capita.png")
task.upload_artifact("CO2 Emissions vs Per Capita", artifact_object="co2_per_capita.png")
plt.show()

# 3. Сравнение стран по выбросам
top_countries = data.groupby('Country')['Kilotons of Co2'].mean().nlargest(10).reset_index()

# Построение графика сравнения стран
plt.figure(figsize=(12, 6))
sns.barplot(x='Kilotons of Co2', y='Country', data=top_countries, palette='viridis')
plt.title("Топ-10 стран с наибольшими выбросами CO₂", fontsize=16)
plt.xlabel("Средние выбросы CO₂ (Kilotons)", fontsize=14)
plt.ylabel("Страна", fontsize=14)
plt.grid(axis='x')
plt.tight_layout()

# Сохранение графика в файл и загрузка в ClearML
plt.savefig("top_10_countries.png")
task.upload_artifact("Top 10 CO2 Emitting Countries", artifact_object="top_10_countries.png")
plt.show()

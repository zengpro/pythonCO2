import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from clearml import Task, Logger, OutputModel
import numpy as np
import joblib
import os

# ClearML: инициализация задачи
task = Task.init(project_name="CO2 Forecasting", task_name="CO2 Model Training")
logger = Logger.current_logger()

# Загрузка данных
def load_data(file_path='datasetCO2.csv'):
    data = pd.read_csv(file_path)
    data['Year'] = pd.to_datetime(data['Date']).dt.year
    return data

data = load_data()

# Выбор признаков и целевой переменной
X = data[['Country', 'Year']]
y = data['Kilotons of Co2']

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Создание трансформеров для категориальных и числовых данных
categorical_features = ['Country']
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

preprocessor = ColumnTransformer(
    transformers=[('cat', categorical_transformer, categorical_features)],
    remainder='passthrough'
)

# Создание модели
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(random_state=42))
])

# Обучение модели
model.fit(X_train, y_train)

# Сохранение модели
model_path = 'co2_model.pkl'
joblib.dump(model, model_path)

# Прогнозирование
y_pred = model.predict(X_test)

# Оценка модели
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Логирование метрик в ClearML
logger.report_scalar("Metrics", "RMSE", iteration=0, value=rmse)
logger.report_scalar("Metrics", "MAE", iteration=0, value=mae)
logger.report_scalar("Metrics", "R2 Score", iteration=0, value=r2)

# Анализ ошибок
errors = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred, 'Error': y_test - y_pred})
errors_path = 'errors.csv'
errors.to_csv(errors_path, index=False)

# Загрузка артефактов в ClearML
task.upload_artifact(name="Trained Model", artifact_object=model_path)
task.upload_artifact(name="Error Analysis", artifact_object=errors_path)

# Вывод результатов
print(f"Модель сохранена в {model_path}.")
print(f"RMSE: {rmse:.2f}, MAE: {mae:.2f}, R2 Score: {r2:.2f}")
print(f"Анализ ошибок сохранён в {errors_path}.")

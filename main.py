from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from clearml import Task, Logger
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd
import joblib

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

# Подготовка данных и создание модели
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd
import joblib

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
    ('poly', PolynomialFeatures(degree=2, include_bias=False))  # Добавляем полиномиальные признаки
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
joblib.dump(model, 'co2_model.pkl')
print("Модель сохранена в 'co2_model.pkl'")

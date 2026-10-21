
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import make_pipeline

colunas = [
    "mpg", "cylinders", "displacement", "horsepower",
    "weight", "acceleration", "model_year", "origin", "car_name"
]

df = pd.read_csv(
    "auto-mpg.data",
    sep=r"\s+",
    header=None,
    names=colunas,
    na_values="?"
)

# Selecionar entradas e saída
x = df.drop(columns=["mpg", "car_name"])
y = df["mpg"]

# Dividir os dados antes do tratamento
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

# Imputar valores ausentes e padronizar entradas
modelo = make_pipeline(
    SimpleImputer(strategy="median"),
    StandardScaler(),
    LinearRegression()
)

modelo.fit(x_train, y_train)

# Prever o consumo no conjunto de teste
y_pred = modelo.predict(x_test)

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MSE: {mse:.4f}")
print(f"RMSE: {np.sqrt(mse):.4f}")
print(f"MAE: {mae:.4f}")
print(f"R²: {r2:.4f}")

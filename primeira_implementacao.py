import pandas as pd

colunas = [
    "mpg",
    "cylinders",
    "displacement",
    "horsepower",
    "weight",
    "acceleration",
    "model_year",
    "origin",
    "car_name"
]

df = pd.read_csv(
    "auto-mpg.data",
    sep=r"\s+",
    header=None,
    names=colunas,
    na_values="?"
)

print(df.head())
print(df.info())
print(df.isna().sum())

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

# Saída desejada
d = df["mpg"].to_numpy(dtype=float)

# Entradas
x = df.drop(columns=["mpg", "car_name"])

colunas_numericas = [
    "cylinders",
    "displacement",
    "horsepower",
    "weight",
    "acceleration",
    "model_year"
]

colunas_categoricas = ["origin"]

# Separar treinamento e teste antes de aprender
# os parâmetros de tratamento.
x_train, x_test, d_train, d_test = train_test_split(
    x, d,
    test_size=0.2,
    random_state=42
)

# Tratar e padronizar as entradas numéricas.
pipeline_numerico = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# Converter a origem em variáveis binárias.
pipeline_categorico = Pipeline([
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessador = ColumnTransformer([
    ("numerico", pipeline_numerico, colunas_numericas),
    ("categorico", pipeline_categorico, colunas_categoricas)
])

# Aprender o tratamento somente com os dados de treino.
x_train = preprocessador.fit_transform(x_train)
x_test = preprocessador.transform(x_test)

# Garantir matrizes NumPy densas.
x_train = np.asarray(x_train, dtype=float)
x_test = np.asarray(x_test, dtype=float)

print("Entradas de treino:", x_train.shape)
print("Entradas de teste:", x_test.shape)
print("Saídas de treino:", d_train.shape)
print("Saídas de teste:", d_test.shape)
print("Primeira amostra:", x_train[0])
print("Consumo real:", d_train[0])
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# Cargar los datos desde la hoja "OLS"
data = pd.read_excel('MLE Datos.xlsx', sheet_name='OLS')

# Supongamos que los datos contienen columnas 'x' y 'y' para regresión
X = data[['X']].values  # Convertimos a un array de 2D para scikit-learn
y = data['Y'].values

# Crear el modelo de regresión lineal
model = LinearRegression()
model.fit(X, y)

# Obtener los parámetros theta_0 y theta_1
theta_0 = model.intercept_
theta_1 = model.coef_[0]

print(f"Parámetro θ0 (intercepto): {theta_0}")
print(f"Parámetro θ1 (pendiente): {theta_1}")

# Verificar la ecuación obtenida
print(f"Ecuación de regresión: y = {theta_0} + {theta_1} * x")

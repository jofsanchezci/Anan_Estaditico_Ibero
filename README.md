
# Análisis Estadístico en Python

Este proyecto contiene ejemplos de cómo realizar un análisis estadístico básico en Python usando librerías populares como `Pandas`, `NumPy`, `SciPy`, y `Scikit-Learn`.

### Conjunto de Datos
Para este análisis, se utiliza un conjunto de datos ficticio de ventas que incluye las siguientes columnas:
- `ventas`: cantidad de productos vendidos.
- `precio`: precio de venta del producto.
- `publicidad`: presupuesto en publicidad para el producto.

### Librerías Necesarias
Para ejecutar este código, necesitas instalar las siguientes librerías:
```bash
pip install pandas numpy scipy matplotlib seaborn scikit-learn
```

### 1. Análisis Descriptivo
Primero, obtenemos estadísticas descriptivas para tener una idea inicial de los datos.

```python
import pandas as pd

data = pd.DataFrame({
    'ventas': [20, 35, 30, 40, 50, 60, 45, 55, 70, 65],
    'precio': [10, 12, 11, 13, 14, 13, 12, 15, 14, 16],
    'publicidad': [200, 250, 230, 270, 290, 310, 280, 300, 320, 340]
})

# Resumen estadístico
descriptivos = data.describe()
print(descriptivos)
```

### 2. Visualización de Datos
Visualización de la relación entre `publicidad` y `ventas`.

```python
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(8, 5))
sns.scatterplot(x='publicidad', y='ventas', data=data)
plt.title("Relación entre Publicidad y Ventas")
plt.xlabel("Presupuesto de Publicidad")
plt.ylabel("Ventas")
plt.show()
```

### 3. Prueba de Hipótesis
Prueba de correlación de Pearson para verificar si existe una relación significativa entre `publicidad` y `ventas`.

```python
from scipy import stats

correlacion, p_valor = stats.pearsonr(data['publicidad'], data['ventas'])
print(f"Correlación: {correlacion}, Valor-p: {p_valor}")

if p_valor < 0.05:
    print("Existe una correlación significativa entre la publicidad y las ventas.")
else:
    print("No hay una correlación significativa entre la publicidad y las ventas.")
```

### 4. Análisis de Regresión Lineal
Aplicación de un modelo de regresión lineal para predecir `ventas` en función de `publicidad`.

```python
from sklearn.linear_model import LinearRegression

X = data[['publicidad']]
y = data['ventas']

modelo = LinearRegression()
modelo.fit(X, y)

# Coeficientes del modelo
print(f"Intersección: {modelo.intercept_}, Coeficiente: {modelo.coef_[0]}")

# Visualización de la regresión
predicciones = modelo.predict(X)
plt.figure(figsize=(8, 5))
plt.scatter(data['publicidad'], data['ventas'], color='blue', label='Datos reales')
plt.plot(data['publicidad'], predicciones, color='red', label='Línea de Regresión')
plt.title("Regresión Lineal entre Publicidad y Ventas")
plt.xlabel("Presupuesto de Publicidad")
plt.ylabel("Ventas")
plt.legend()
plt.show()
```

### 5. Prueba T para Comparar Promedios
Prueba T para determinar si existe una diferencia significativa en las `ventas` según si el presupuesto en `publicidad` es alto o bajo.

```python
ventas_alta_publicidad = data[data['publicidad'] > 300]['ventas']
ventas_baja_publicidad = data[data['publicidad'] <= 300]['ventas']

t_stat, p_valor_t = stats.ttest_ind(ventas_alta_publicidad, ventas_baja_publicidad)
print(f"Estadístico t: {t_stat}, Valor-p: {p_valor_t}")

if p_valor_t < 0.05:
    print("Existe una diferencia significativa en las ventas entre los dos grupos.")
else:
    print("No hay una diferencia significativa en las ventas entre los dos grupos.")
```

### Conclusión
Este archivo README demuestra cómo realizar un análisis estadístico básico, incluyendo estadística descriptiva, prueba de hipótesis, análisis de correlación y regresión lineal en Python.

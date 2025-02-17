import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from collections import Counter

# Cargamos los datos del archivo de los pilotos
archivo = "DatosPilotos.csv"
df = pd.read_csv(archivo)

df.columns = df.columns.str.strip()  # Elimina espacios antes y después de los nombres
df = df.drop('Porcentaje_Abandonos', axis=1)


# Corregir posibles errores en nombres y duplicados
df["Driver"] = df["Driver"].str.strip().str.upper()
df = df.drop_duplicates()

# Separar características y etiquetas
X = df.iloc[:, 1:].values  # Todas las columnas excepto el nombre
y = df.iloc[:, 0].values   # Primera columna: nombres de los pilotos

# Normalizar características
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Codificar etiquetas (pilotos)
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# Definir el modelo de red neuronal
model = Sequential([
    Dense(16, activation='relu', input_shape=(X.shape[1],)),   
    Dense(8, activation='relu'),    
    Dense(len(np.unique(y_encoded)), activation='softmax')  
])

# Compilar el modelo
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Entrenar el modelo
model.fit(X, y_encoded, epochs=100, batch_size=4, validation_split=0.2)

# Función para simular podio
def simulate_podium(model, X, encoder):
    predictions = model.predict(X)  # Obtener todas las predicciones
    podium_indices = np.argsort(predictions, axis=1)[:, -3:][:, ::-1]  # Top 3 por fila, en orden descendente
    # Convertimos cada fila en nombres de pilotos
    podium_results = [encoder.inverse_transform(row) for row in podium_indices]
    return podium_results


# Inicializar contadores para la tabla de frecuencias
first_place = Counter()
second_place = Counter()
third_place = Counter()

# Ejecutar la simulación sobre todos los pilotos
podium_results = simulate_podium(model, X, encoder)
for podium in podium_results:
    first_place[podium[2]] += 1
    second_place[podium[1]] += 1
    third_place[podium[0]] += 1

# Crear tabla de frecuencias
table = pd.DataFrame({'1° Lugar': first_place, '2° Lugar': second_place, '3° Lugar': third_place}).fillna(0)

# Calcular puntos
table['Puntos'] = (table['1° Lugar'] * 5) + (table['2° Lugar'] * 3) + (table['3° Lugar'] * 1)

# Ordenar por puntos
table = table.sort_values(by='Puntos', ascending=False)

print("\n🏆 Tabla de Frecuencias 🏆")
print(table)



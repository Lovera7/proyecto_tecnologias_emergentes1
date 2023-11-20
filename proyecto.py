import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv('heart_attack_prediction_dataset.csv')
df.head(10)

df.isna().any()

df.count()

df = df.drop(columns=['Patient ID', 'Country', 'Continent', 'Hemisphere', 'Income'])
df.head(10)

df.isna().sum()

fig = plt.figure(figsize=(20, 20))
ax = fig.gca()
df.hist(ax=ax)

df.dtypes

# Dividir la columna en dos
df[['Systolic BP', 'Diastolic BP']] = df['Blood Pressure'].str.split('/', expand=True)

# Convertir a numérico
df['Systolic BP'] = pd.to_numeric(df['Systolic BP'], errors='coerce')
df['Diastolic BP'] = pd.to_numeric(df['Diastolic BP'], errors='coerce')

# Eliminar la columna original "Blood Pressure"
df.drop(columns=['Blood Pressure'], inplace=True)

df.head(10)

df = pd.get_dummies(df, columns=['Sex', 'Diet'], drop_first=True)

df['Sex_Male'] = df['Sex_Male'].astype(int)
df['Diet_Healthy'] = df['Diet_Healthy'].astype(int)
df['Diet_Unhealthy'] = df['Diet_Unhealthy'].astype(int)

df['Exercise Hours Per Week'] = df['Exercise Hours Per Week'].round().astype('int')
df['Sedentary Hours Per Day'] = df['Sedentary Hours Per Day'].round().astype('int')

# Mover 'Heart Attack Risk' al final
df = df[[col for col in df.columns if col != 'Heart Attack Risk'] + ['Heart Attack Risk']]

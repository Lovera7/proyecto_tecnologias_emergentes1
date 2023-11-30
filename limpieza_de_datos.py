import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# !wget https://raw.githubusercontent.com/Lovera7/proyecto_tecnologias_emergentes1/main/heart_attack_prediction_dataset.csv

'''
Patient ID - Unique identifier for each patient
Age - Age of the patient
Sex - Gender of the patient (Male/Female)
Cholesterol - Cholesterol levels of the patient
Blood Pressure - Blood pressure of the patient (systolic/diastolic)
Heart Rate - Heart rate of the patient
Diabetes - Whether the patient has diabetes (Yes/No)
Family History - Family history of heart-related problems (1: Yes, 0: No)
Smoking - Smoking status of the patient (1: Smoker, 0: Non-smoker)
Obesity - Obesity status of the patient (1: Obese, 0: Not obese)
Alcohol Consumption - Level of alcohol consumption by the patient (None/Light/Moderate/Heavy)
Exercise Hours Per Week - Number of exercise hours per week
Diet - Dietary habits of the patient (Healthy/Average/Unhealthy)
Previous Heart Problems - Previous heart problems of the patient (1: Yes, 0: No)
Medication Use - Medication usage by the patient (1: Yes, 0: No)
Stress Level - Stress level reported by the patient (1-10)
Sedentary Hours Per Day - Hours of sedentary activity per day
Income - Income level of the patient
BMI - Body Mass Index (BMI) of the patient
Triglycerides - Triglyceride levels of the patient
Physical Activity Days Per Week - Days of physical activity per week
Sleep Hours Per Day - Hours of sleep per day
Country - Country of the patient
Continent - Continent where the patient resides
Hemisphere - Hemisphere where the patient resides
Heart Attack Risk - Presence of heart attack risk (1: Yes, 0: No)
'''


df = pd.read_csv('heart_attack_prediction_dataset.csv')
df.head(10)

df.dtypes

# Verificar si hay valores nulos
df.isnull().sum()

df = df.drop(columns=['Patient ID', 'Country', 'Continent', 'Hemisphere', 'Income'])

# Dividir la columna 'Blood Pressure' en 'Systolic_BP' y 'Diastolic_BP'
df[['Systolic_BP', 'Diastolic_BP']] = df['Blood Pressure'].str.split('/', expand=True)

# Convertir las nuevas columnas a tipo numérico
df['Systolic_BP'] = pd.to_numeric(df['Systolic_BP'], errors='coerce')
df['Diastolic_BP'] = pd.to_numeric(df['Diastolic_BP'], errors='coerce')

# Eliminar la columna original 'Blood Pressure'
df = df.drop('Blood Pressure', axis=1)

df = pd.get_dummies(df, columns=['Diet'])

df = pd.get_dummies(df, columns=['Sex'])

# Analisis exploratorio
df.describe()
df.isnull().sum()

import seaborn as sns

df.hist(bins=50, figsize=(20,15))
plt.show()

corr_matrix = df.corr()
print(corr_matrix["Heart Attack Risk"].sort_values(ascending=False))

plt.figure(figsize=(10,10))
sns.heatmap(corr_matrix, annot=True, cmap="YlGnBu")
plt.show()

sns.countplot(x='Diet_Healthy', data=df)
plt.show()

'''
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier

# Dividir el dataset en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(df.iloc[:, 1:], df.iloc[:, 0], test_size=0.3, shuffle=True, random_state=42)
# Crear un objeto GradientBoostingClassifier
clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=42)

# Entrenar el modelo de Gradient Boosting con los datos de entrenamiento
clf.fit(X_train, y_train)

# Predecir las etiquetas de clase para el conjunto de prueba
y_pred = clf.predict(X_test)

# Evaluar el modelo
score = accuracy_score(y_test, y_pred)
print('Accuracy:', score)

cm = confusion_matrix(y_test, y_pred)
print('Confusion Matrix:\n', cm)
'''

# Revisar valores atipicos
# Python
Q1 = df['Age'].quantile(0.25)
Q3 = df['Age'].quantile(0.75)
IQR = Q3 - Q1

# Valores atípicos
outliers = df[(df['Age'] < Q1 - 1.5 * IQR) | (df['Age'] > Q3 + 1.5 * IQR)]
print(outliers)

# Python
Q1 = df['Cholesterol'].quantile(0.25)
Q3 = df['Cholesterol'].quantile(0.75)
IQR = Q3 - Q1

# Valores atípicos
outliers = df[(df['Cholesterol'] < Q1 - 1.5 * IQR) | (df['Cholesterol'] > Q3 + 1.5 * IQR)]
print(outliers)


# Python
Q1 = df['Heart Rate'].quantile(0.25)
Q3 = df['Heart Rate'].quantile(0.75)
IQR = Q3 - Q1

# Valores atípicos
outliers = df[(df['Heart Rate'] < Q1 - 1.5 * IQR) | (df['Heart Rate'] > Q3 + 1.5 * IQR)]
print(outliers)

# Python
Q1 = df['Exercise Hours Per Week'].quantile(0.25)
Q3 = df['Exercise Hours Per Week'].quantile(0.75)
IQR = Q3 - Q1

# Valores atípicos
outliers = df[(df['Exercise Hours Per Week'] < Q1 - 1.5 * IQR) | (df['Exercise Hours Per Week'] > Q3 + 1.5 * IQR)]
print(outliers)


# Python
Q1 = df['Stress Level'].quantile(0.25)
Q3 = df['Stress Level'].quantile(0.75)
IQR = Q3 - Q1

# Valores atípicos
outliers = df[(df['Stress Level'] < Q1 - 1.5 * IQR) | (df['Stress Level'] > Q3 + 1.5 * IQR)]
print(outliers)

# Python
Q1 = df['Sedentary Hours Per Day'].quantile(0.25)
Q3 = df['Sedentary Hours Per Day'].quantile(0.75)
IQR = Q3 - Q1

# Valores atípicos
outliers = df[(df['Sedentary Hours Per Day'] < Q1 - 1.5 * IQR) | (df['Sedentary Hours Per Day'] > Q3 + 1.5 * IQR)]
print(outliers)

# Python
Q1 = df['BMI'].quantile(0.25)
Q3 = df['BMI'].quantile(0.75)
IQR = Q3 - Q1

# Valores atípicos
outliers = df[(df['BMI'] < Q1 - 1.5 * IQR) | (df['BMI'] > Q3 + 1.5 * IQR)]
print(outliers)

# Python
Q1 = df['Triglycerides'].quantile(0.25)
Q3 = df['Triglycerides'].quantile(0.75)
IQR = Q3 - Q1

# Valores atípicos
outliers = df[(df['Triglycerides'] < Q1 - 1.5 * IQR) | (df['Triglycerides'] > Q3 + 1.5 * IQR)]
print(outliers)


# Python
Q1 = df['Physical Activity Days Per Week'].quantile(0.25)
Q3 = df['Physical Activity Days Per Week'].quantile(0.75)
IQR = Q3 - Q1

# Valores atípicos
outliers = df[(df['Physical Activity Days Per Week'] < Q1 - 1.5 * IQR) | (df['Physical Activity Days Per Week'] > Q3 + 1.5 * IQR)]
print(outliers)


# Python
Q1 = df['Sleep Hours Per Day'].quantile(0.25)
Q3 = df['Sleep Hours Per Day'].quantile(0.75)
IQR = Q3 - Q1

# Valores atípicos
outliers = df[(df['Sleep Hours Per Day'] < Q1 - 1.5 * IQR) | (df['Sleep Hours Per Day'] > Q3 + 1.5 * IQR)]
print(outliers)


# Python
Q1 = df['Systolic_BP'].quantile(0.25)
Q3 = df['Systolic_BP'].quantile(0.75)
IQR = Q3 - Q1

# Valores atípicos
outliers = df[(df['Systolic_BP'] < Q1 - 1.5 * IQR) | (df['Systolic_BP'] > Q3 + 1.5 * IQR)]
print(outliers)

# Python
Q1 = df['Diastolic_BP'].quantile(0.25)
Q3 = df['Diastolic_BP'].quantile(0.75)
IQR = Q3 - Q1

# Valores atípicos
outliers = df[(df['Diastolic_BP'] < Q1 - 1.5 * IQR) | (df['Diastolic_BP'] > Q3 + 1.5 * IQR)]
print(outliers)


# Analisis exploratorio

fig, axs = plt.subplots(3, 3, figsize=(15, 15))

# Histograma para 'Age'
axs[0, 0].hist(df['Age'], bins=8, edgecolor='black')
axs[0, 0].set_title('Distribución de la Edad')

# Histograma para 'Cholesterol'
axs[0, 1].hist(df['Cholesterol'], bins=6, edgecolor='black')
axs[0, 1].set_title('Distribución del Colesterol')

# Histograma para 'Heart Rate'
axs[0, 2].hist(df['Heart Rate'], bins=8, edgecolor='black')
axs[0, 2].set_title('Distribución de la Frecuencia Cardíaca')

# Histograma para 'BMI'
axs[1, 0].hist(df['BMI'], bins=5, edgecolor='black')
axs[1, 0].set_title('Distribución del Índice de Masa Corporal')

# Histograma para 'Triglycerides'
axs[1, 1].hist(df['Triglycerides'], bins=8, edgecolor='black')
axs[1, 1].set_title('Distribución de los Triglicéridos')

# Histograma para 'Systolic_BP'
axs[1, 2].hist(df['Systolic_BP'], bins=5, edgecolor='black')
axs[1, 2].set_title('Distribución de la Presión Arterial Sistólica')

# Histograma para 'Diastolic_BP'
axs[2, 0].hist(df['Diastolic_BP'], bins=6, edgecolor='black')
axs[2, 0].set_title('Distribución de la Presión Arterial Diastólica')

plt.tight_layout()
plt.show()



from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier

# Dividir el dataset en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(df.iloc[:, 1:], df.iloc[:, 0], test_size=0.3, shuffle=True, random_state=42)
clf = GradientBoostingClassifier(n_estimators=50, learning_rate=0.1, max_depth=1, random_state=42)
# Entrenar el modelo
# Lista de características para eliminar
features_to_drop = ['Previous Heart Problems', 'Medication Use', 'Alcohol Consumption', 'Obesity', 'Smoking', 'Family History', 'Diabetes', 'Diet_Average', 'Diet_Healthy', 'Sex_Female', 'Sex_Male']

# Eliminar las características
X_train = X_train.drop(features_to_drop, axis=1)
# X_train = X_train.drop('Age_Categories', axis=1)
clf.fit(X_train, y_train)

# Obtener la importancia de las características
feature_importances = clf.feature_importances_

# Crear un DataFrame con las características y su importancia
importance_df = pd.DataFrame({'feature': X_train.columns, 'importance': feature_importances})

# Ordenar el DataFrame por importancia de la característica
importance_df = importance_df.sort_values('importance', ascending=False)

print(importance_df)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Predecir las etiquetas para el conjunto de prueba
y_pred = clf.predict(X_test.drop(features_to_drop, axis=1))

# Calcular y mostrar las métricas
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))


from sklearn.metrics import confusion_matrix

# Calcular y mostrar la matriz de confusión
print(confusion_matrix(y_test, y_pred))


from sklearn.model_selection import GridSearchCV

# Definir los parámetros para la búsqueda de cuadrícula
param_grid = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 1.0],
    'max_depth': [1, 2, 3]
}

# Crear el objeto de búsqueda de cuadrícula
grid_search = GridSearchCV(GradientBoostingClassifier(random_state=42), param_grid, cv=5)

# Realizar la búsqueda de cuadrícula
grid_search.fit(X_train, y_train)

# Imprimir los mejores parámetros
print(grid_search.best_params_)
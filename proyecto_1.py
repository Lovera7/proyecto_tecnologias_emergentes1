import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.manifold import TSNE
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

df = pd.read_csv('heart_attack_prediction_dataset.csv')
df.head(10)

# LIMPIEZA DE DATOS
df.dtypes
df.isna().sum()
df.info()
df.duplicated().sum() # Identificar duplicados

# ANALISIS EXPLORATORIO
# Ejemplo con la variable 'Age'
plt.figure(figsize=(8, 6))
sns.histplot(df['Age'], bins=9, kde=True, color='skyblue')
plt.title('Distribución de Edades')
plt.xlabel('Edad')
plt.ylabel('Frecuencia')
plt.show()

# Ejemplo con la variable 'Cholesterol'
plt.figure(figsize=(8, 6))
sns.boxplot(x='Cholesterol', data=df, palette='Set2')
plt.title('Distribución de Colesterol')
plt.xlabel('Colesterol')
plt.show()

# Ejemplo con la variable 'Sex'
plt.figure(figsize=(8, 6))
sns.countplot(x='Sex', data=df, palette='pastel')
plt.title('Distribución de Género')
plt.xlabel('Género')
plt.ylabel('Frecuencia')
plt.show()

num_features = df.select_dtypes(include=[np.number]).columns.drop('Heart Attack Risk')

for feature in num_features:
    plt.figure(figsize=(10, 4))
    sns.kdeplot(data=df, x=feature, hue="Heart Attack Risk", common_norm=False)
    plt.show()
    

# INGENIERIA DE FEATURES
# Dividir la columna 'Blood Pressure' en dos columnas
df[['Systolic_BP', 'Diastolic_BP']] = df['Blood Pressure'].str.split('/', expand=True)

# Convertir las nuevas columnas a numéricas
df['Systolic_BP'] = pd.to_numeric(df['Systolic_BP'], errors='coerce')
df['Diastolic_BP'] = pd.to_numeric(df['Diastolic_BP'], errors='coerce')

# Eliminar la columna original 'Blood Pressure'
df = df.drop(columns='Blood Pressure')
# Mover 'tu_variable_objetivo' al inicio
df = df[['Heart Attack Risk'] + [col for col in df.columns if col != 'Heart Attack Risk']]

# Calcular la matriz de correlación
corr_matrix = df.corr()

# Crear una máscara para el triángulo superior
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

# Configurar la figura de matplotlib
plt.figure(figsize=(20, 20))

# Dibujar el mapa de calor
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f", cmap='coolwarm', cbar=True, square=True)

# Mostrar la gráfica
plt.show()

# PREPROCESAMIENTO DE DATOS
df = df.drop(columns=['Patient ID', 'Country', 'Continent', 'Hemisphere', 'Income', 'Physical Activity Days Per Week', 'Sedentary Hours Per Day'])

# Convertir la columna 'Sex' a numérica con codificación One-Hot
df = pd.get_dummies(df, columns=['Sex'], prefix = ['Sex'])
# Convertir la columna 'Diet' a numérica con codificación ordinal
diet_mapping = {'Unhealthy': 0, 'Average': 1, 'Healthy': 2}
df['Diet'] = df['Diet'].map(diet_mapping)


# Lista de columnas a normalizar
cols_to_normalize = ['Age', 'Cholesterol', 'Heart Rate', 'Exercise Hours Per Week', 'Stress Level', 'BMI', 'Triglycerides', 'Sleep Hours Per Day', 'Systolic_BP', 'Diastolic_BP']

# Normalización
scaler = MinMaxScaler()
df[cols_to_normalize] = scaler.fit_transform(df[cols_to_normalize])
df.head()

# Aplicar t-SNE
tsne = TSNE(n_components=3, n_iter=2000, random_state=42)
tsne_results = tsne.fit_transform(df)

# Crear un DataFrame con los resultados de t-SNE
df_tsne = pd.DataFrame(data=tsne_results, columns=['Component 1', 'Component 2', 'Component 3'])

# Crear un DataFrame con los resultados de t-SNE y la etiqueta de riesgo de ataque al corazón
df_tsne = pd.concat([df_tsne, df['Heart Attack Risk'].reset_index(drop=True)], axis=1)

# Crear un gráfico de dispersión 3D interactivo
fig = px.scatter_3d(df_tsne, x='Component 1', y='Component 2', z='Component 3', color='Heart Attack Risk')

# Añadir etiquetas a los ejes y un título al gráfico
fig.update_layout(
    title='3 component t-SNE',
    scene = dict(
        xaxis_title='Component 1',
        yaxis_title='Component 2',
        zaxis_title='Component 3'
    ),
    legend_title='Heart Attack Risk',
    autosize=False,
    width=1000,  # Ancho del gráfico en píxeles
    height=1000,  # Altura del gráfico en píxeles
)

fig.update_traces(marker_line_color = 'black',
                  marker_line_width = 2)

# Mostrar el gráfico
fig.show()


# Aplicar SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(df.iloc[:, 1:], df.iloc[:, 0])

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, shuffle=True, random_state=42)


# Crear y entrenar el modelo de Bosques Aleatorios
rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_split=2, min_samples_leaf=1, random_state=42)
rf_model.fit(X_train, y_train)

# Crear y entrenar un modelo de XGBoost
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', reg_alpha=0.1, reg_lambda=1.0, random_state=42)
xgb_model.fit(X_train, y_train)

# Inicializar el modelo
clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42)
clf.fit(X_train, y_train)


# Función para calcular las métricas y agregarlas al DataFrame
def calculate_metrics(model, X_train, y_train, X_test, y_test):
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)

    train_precision = precision_score(y_train, y_train_pred)
    test_precision = precision_score(y_test, y_test_pred)

    train_recall = recall_score(y_train, y_train_pred)
    test_recall = recall_score(y_test, y_test_pred)

    train_f1 = f1_score(y_train, y_train_pred)
    test_f1 = f1_score(y_test, y_test_pred)

    train_roc_auc = roc_auc_score(y_train, model.predict_proba(X_train)[:, 1])
    test_roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

    return pd.DataFrame({
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC Score'],
        'Train Score': [train_accuracy, train_precision, train_recall, train_f1, train_roc_auc],
        'Test Score': [test_accuracy, test_precision, test_recall, test_f1, test_roc_auc]
    })
    
    
# Calcular las métricas para cada modelo
results_df1 = calculate_metrics(rf_model, X_train, y_train, X_test, y_test)
results_df2 = calculate_metrics(xgb_model, X_train, y_train, X_test, y_test)
results_df3 = calculate_metrics(clf, X_train, y_train, X_test, y_test)

# Concatenar los resultados en un solo DataFrame
results_df = pd.concat([results_df1, results_df2, results_df3], keys=['Bosques Aleatorios', 'XGBoost', 'Gradient Boosting'])

# Imprimir los resultados
results_df


# Predicciones
y_pred_rf = rf_model.predict(X_test)
y_pred_xgb = xgb_model.predict(X_test)
y_pred_clf = clf.predict(X_test)

# Matrices de confusión
cm_rf = confusion_matrix(y_test, y_pred_rf)
cm_xgb = confusion_matrix(y_test, y_pred_xgb)
cm_clf = confusion_matrix(y_test, y_pred_clf)

# Visualizar las matrices de confusión
plt.figure(figsize=(18, 4))

plt.subplot(131)
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues')
plt.title('Random Forest Confusion Matrix')

plt.subplot(132)
sns.heatmap(cm_xgb, annot=True, fmt='d', cmap='Blues')
plt.title('XGBoost Confusion Matrix')

plt.subplot(133)
sns.heatmap(cm_clf, annot=True, fmt='d', cmap='Blues')
plt.title('Gradient Boosting Confusion Matrix')

plt.show()


# Calcular los puntajes de validación cruzada
rf_scores = cross_val_score(rf_model, X_train, y_train, cv=5)
xgb_scores = cross_val_score(xgb_model, X_train, y_train, cv=5)
clf_scores = cross_val_score(clf, X_train, y_train, cv=5)

# Crear un DataFrame para almacenar los resultados
df_scores = pd.DataFrame({
    'Random Forest': rf_scores,
    'XGBoost': xgb_scores,
    'Gradient Boosting': clf_scores
})

# Transponer el DataFrame para que cada fila corresponda a un modelo
df_scores = df_scores.transpose()

# Agregar una columna para el promedio de los puntajes
df_scores['Promedio'] = df_scores.mean(axis=1)

# Mostrar el DataFrame
print(df_scores)
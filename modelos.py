from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from imblearn.over_sampling import SMOTE

# Dividir el dataset en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(df.iloc[:, 1:], df.iloc[:, 0], test_size=0.3, shuffle=True, random_state=42)

clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=1, random_state=42)

# Crear el objeto SMOTE
smote = SMOTE(random_state=42)

# Aplicar SMOTE a los datos de entrenamiento
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# Ahora puedes entrenar tu modelo con los datos balanceados
clf.fit(X_train_res, y_train_res)

# Predecir las etiquetas para el conjunto de prueba
y_pred = clf.predict(X_test)

# Calcular y mostrar las métricas
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
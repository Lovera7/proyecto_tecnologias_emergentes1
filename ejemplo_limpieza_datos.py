import pandas as pd

sales = pd.read_csv('sales.csv')
sales.head(10)

# CAMBIAR TIPOS DE DATOS SI ES NECESARIO
# 1 Verificar los tipos de datos de nuestro dataset - para saber si hay que convertirlos
sales.dtypes
sales.info()

sales['Revenue'] = sales['Revenue'].str.strip('$')
sales['Revenue'] = sales['Revenue'].astype('int')

# Verificar que Revenue ya es int - devuelve un error si no es entero y continua si es entero
assert sales['Revenue'].dtype == 'int'

# Convertir una columna a tipo de datos category
sales['Product'] = sales['Product'].astype('category')
sales['Product'].describe()

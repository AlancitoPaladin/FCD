import pandas as pd

# Cargar CSV
df = pd.read_csv("../u1/features.csv")

# Ver las primeras filas
print(df.head())

# Ver solo algunas columnas para no saturar la pantalla
print(df.iloc[:, :10])  # primeras 10 features + etiqueta

import pandas as pd

# df = tu dataframe original

# Asegurar que volumen_mxn sea numérico
df["volumen_mxn"] = pd.to_numeric(df["volumen_mxn"], errors="coerce")

# -------------------------------------------------------
# 1. Identificar clientes con más de un producto/clasificación
#    dentro de cada mes y zona
# -------------------------------------------------------

productos_cliente = (
    df.groupby(["mes", "zona", "nombre"])["clasificacion"]
      .nunique()
      .reset_index(name="num_productos")
)

productos_cliente["mas_de_un_producto"] = productos_cliente["num_productos"] > 1

# Agregar esta marca al dataframe original
df2 = df.merge(
    productos_cliente,
    on=["mes", "zona", "nombre"],
    how="left"
)

# -------------------------------------------------------
# 2. Tabla principal por mes, zona y clasificación
# -------------------------------------------------------

tabla = (
    df2.groupby(["mes", "zona", "clasificacion"])
       .agg(
           clientes=("nombre", "nunique"),
           volumen_promedio=("volumen_mxn", "mean"),
           clientes_mas_de_un_producto=("mas_de_un_producto", "sum")
       )
       .reset_index()
)

# Ojo: clientes_mas_de_un_producto puede estar duplicado si un cliente
# aparece varias veces dentro de la misma clasificación.
# Para evitar eso, usamos una versión deduplicada:

tabla_multi_producto = (
    df2.drop_duplicates(["mes", "zona", "clasificacion", "nombre"])
       .groupby(["mes", "zona", "clasificacion"])
       .agg(
           clientes_mas_de_un_producto=("mas_de_un_producto", "sum")
       )
       .reset_index()
)

tabla_base = (
    df2.groupby(["mes", "zona", "clasificacion"])
       .agg(
           clientes=("nombre", "nunique"),
           volumen_promedio=("volumen_mxn", "mean")
       )
       .reset_index()
)

tabla = tabla_base.merge(
    tabla_multi_producto,
    on=["mes", "zona", "clasificacion"],
    how="left"
)

# -------------------------------------------------------
# 3. Pivotear para que la clasificación quede como columnas
# -------------------------------------------------------

tabla_pivot = tabla.pivot_table(
    index=["mes", "zona"],
    columns="clasificacion",
    values=["clientes", "volumen_promedio", "clientes_mas_de_un_producto"],
    aggfunc="sum"
)

# Aplanar nombres de columnas
tabla_pivot.columns = [
    f"{metrica}_{clasificacion}"
    for metrica, clasificacion in tabla_pivot.columns
]

tabla_pivot = tabla_pivot.reset_index()

# Opcional: llenar nulos con cero
tabla_pivot = tabla_pivot.fillna(0)

tabla_pivot

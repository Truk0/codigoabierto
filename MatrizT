import pandas as pd
import numpy as np

archivo = "Info_matrices.xlsx"
hoja_origen = "Base_Dic'25"   # mes inicial
hoja_destino = "Base_Ene'25"  # mes final

def clean_sheet(df):
    # Si la primera fila trae headers dentro de la tabla, úsala como encabezado
    first = df.iloc[0].astype(str).str.strip().str.upper().tolist()
    if "CLIENTE" in first and "CALIFICACIÓN" in first:
        df.columns = df.iloc[0]
        df = df.iloc[1:].reset_index(drop=True)

    df.columns = [str(c).strip().upper() for c in df.columns]

    col_cliente = next((c for c in df.columns if "CLIENTE" in c), None)
    col_calif   = next((c for c in df.columns if "CALIFIC" in c), None)

    out = df[[col_cliente, col_calif]].copy()
    out.columns = ["CLIENTE", "CALIFICACION"]

    out["CLIENTE"] = pd.to_numeric(out["CLIENTE"], errors="coerce").astype("Int64")
    out["CALIFICACION"] = out["CALIFICACION"].astype(str).str.strip().str.upper()

    out = out.dropna(subset=["CLIENTE"])
    out = out[out["CALIFICACION"].ne("NAN")]
    return out

def collapse_one_rating(df):
    # 1 calificación por cliente: toma la más frecuente; si hay empate, toma la última observada
    def pick(series):
        s = series.dropna()
        if s.empty:
            return np.nan
        vc = s.value_counts()
        top = vc[vc == vc.max()].index.tolist()
        if len(top) == 1:
            return top[0]
        for val in reversed(series.tolist()):
            if val in top:
                return val
        return top[0]

    return df.groupby("CLIENTE", as_index=False)["CALIFICACION"].agg(pick)

def transition_matrix(df_from, df_to, missing_label="SIN INFO", include_missing=True):
    merged = df_from.merge(
        df_to,
        on="CLIENTE",
        how="outer" if include_missing else "inner",
        suffixes=("_FROM", "_TO")
    )
    merged["FROM"] = merged["CALIFICACION_FROM"].fillna(missing_label)
    merged["TO"]   = merged["CALIFICACION_TO"].fillna(missing_label)

    mat = pd.crosstab(merged["FROM"], merged["TO"]).astype(int)
    row_pct = (mat.div(mat.sum(axis=1), axis=0) * 100).round(2)
    return merged, mat, row_pct

# Leer
df_from_raw = pd.read_excel(archivo, sheet_name=hoja_origen)
df_to_raw   = pd.read_excel(archivo, sheet_name=hoja_destino)

# Limpiar y dejar 1 rating por cliente
df_from = collapse_one_rating(clean_sheet(df_from_raw))
df_to   = collapse_one_rating(clean_sheet(df_to_raw))

# Matriz
detalle, matriz_conteos, matriz_pct = transition_matrix(df_from, df_to)

# Guardar
salida = "matriz_transicion.xlsx"
with pd.ExcelWriter(salida, engine="openpyxl") as writer:
    matriz_conteos.to_excel(writer, sheet_name="Conteos")
    matriz_pct.to_excel(writer, sheet_name="Porcentaje_fila")
    detalle.to_excel(writer, sheet_name="Detalle_merge", index=False)

print("Listo ->", salida)

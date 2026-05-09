import pandas as pd

def convertir_tabla(df):
    df = df.copy()

    # Fechas
    df["fecha_operacion"] = pd.to_datetime(df["fecha_operacion"], dayfirst=True)
    df["fecha_liquidacion"] = pd.to_datetime(df["fecha_liquidacion"], dayfirst=True)

    # Ventas negativas, compras positivas
    df["volumen_signado"] = df["volumen_usd"].where(
        df["tipo_operacion"].eq("C"),
        -df["volumen_usd"]
    )

    # Tabla pivote base
    tabla = (
        df.pivot_table(
            index=["fecha_operacion", "tipo_operacion"],
            columns="fecha_liquidacion",
            values="volumen_signado",
            aggfunc="sum",
            fill_value=0
        )
        .sort_index()
    )

    # Ordenar tipo de operación: C primero, V después
    tabla = tabla.reindex(
        pd.MultiIndex.from_product(
            [
                sorted(df["fecha_operacion"].unique()),
                ["C", "V"]
            ],
            names=["fecha_operacion", "tipo_operacion"]
        )
    ).fillna(0)

    # Balance por fecha de liquidación
    balance = tabla.sum(axis=0)

    # En tu ejemplo el acumulado empieza desde la segunda columna visible
    acumulado = balance.cumsum()

    # Agregar filas finales
    tabla.loc[("", "Balance"), :] = balance
    tabla.loc[("", "Acumulado"), :] = acumulado

    # Formato de fechas para mostrar como dd/mm/yy
    tabla.columns = [c.strftime("%d/%m/%y") for c in tabla.columns]

    tabla = tabla.reset_index()

    tabla["fecha_operacion"] = tabla["fecha_operacion"].apply(
        lambda x: x.strftime("%d/%m/%y") if hasattr(x, "strftime") else x
    )

    return tabla

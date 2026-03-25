import pandas as pd
import numpy as np

# =========================
# CONFIGURACIÓN DE MESES
# =========================
MESES_ORDEN = [
    "Diciembre 2024",
    "Enero 2025", "Febrero 2025", "Marzo 2025", "Abril 2025",
    "Mayo 2025", "Junio 2025", "Julio 2025", "Agosto 2025",
    "Septiembre 2025", "Octubre 2025", "Noviembre 2025", "Diciembre 2025"
]

N_TRANSICIONES = len(MESES_ORDEN) - 1  # 12


def matriz_transicion_ponderada(df):
    """
    Construye una matriz de transición ponderada con estas reglas:
    1) Se toman solo los Id que existen en Diciembre 2024.
    2) Cada Id aporta según hasta qué mes aparece.
    3) Cada transición mensual observada aporta 1/12.
    4) La matriz final puede verse en conteos ponderados y en probabilidades por renglón.
    
    Parámetros
    ----------
    df : pd.DataFrame
        Debe tener columnas: ['Id', 'Calif', 'Mes-Año']
    
    Retorna
    -------
    transiciones_detalle : pd.DataFrame
        Detalle de cada transición usada.
    matriz_ponderada : pd.DataFrame
        Matriz de transición en pesos acumulados.
    matriz_prob : pd.DataFrame
        Matriz normalizada por renglón (probabilidades).
    pesos_por_id : pd.DataFrame
        Resumen del peso total aportado por cada Id.
    """

    df = df.copy()

    # Validación mínima
    cols_esperadas = {'Id', 'Calif', 'Mes-Año'}
    faltantes = cols_esperadas - set(df.columns)
    if faltantes:
        raise ValueError(f"Faltan columnas requeridas: {faltantes}")

    # Limpiar texto
    df['Mes-Año'] = df['Mes-Año'].astype(str).str.strip()
    df['Calif'] = df['Calif'].astype(str).str.strip()

    # Mantener solo meses válidos
    df = df[df['Mes-Año'].isin(MESES_ORDEN)].copy()

    # Orden temporal
    orden_mes = {mes: i for i, mes in enumerate(MESES_ORDEN)}
    df['mes_idx'] = df['Mes-Año'].map(orden_mes)

    # Base: Ids presentes en Diciembre 2024
    ids_base = set(df.loc[df['Mes-Año'] == 'Diciembre 2024', 'Id'].unique())
    df = df[df['Id'].isin(ids_base)].copy()

    # Si hubiera duplicados por Id-Mes, se toma el primero
    # (si prefieres otra regla, se puede cambiar)
    df = df.sort_values(['Id', 'mes_idx']).drop_duplicates(
        subset=['Id', 'mes_idx'], keep='first'
    )

    # Pasar a formato ancho: una columna por mes
    panel = df.pivot(index='Id', columns='mes_idx', values='Calif')

    # Asegurar que existan todas las columnas 0..12
    panel = panel.reindex(columns=range(len(MESES_ORDEN)))

    detalles = []
    resumen_pesos = []

    for id_ in panel.index:
        serie = panel.loc[id_]

        # Solo usamos ids con calificación base en Diciembre 2024
        if pd.isna(serie.loc[0]):
            continue

        # Encontrar hasta dónde aparece de forma consecutiva desde Dic 2024
        ultimo_consecutivo = 0
        for j in range(1, len(MESES_ORDEN)):
            if pd.notna(serie.loc[j]):
                ultimo_consecutivo = j
            else:
                break

        # Número de transiciones observadas
        n_obs = ultimo_consecutivo  # porque inicia en 0
        peso_total_id = n_obs / N_TRANSICIONES if N_TRANSICIONES > 0 else 0

        resumen_pesos.append({
            'Id': id_,
            'ultimo_mes_observado': MESES_ORDEN[ultimo_consecutivo],
            'transiciones_observadas': n_obs,
            'peso_total_id': peso_total_id
        })

        # Cada transición mensual observada pesa 1/12
        for j in range(ultimo_consecutivo):
            calif_origen = serie.loc[j]
            calif_destino = serie.loc[j + 1]

            if pd.notna(calif_origen) and pd.notna(calif_destino):
                detalles.append({
                    'Id': id_,
                    'mes_origen': MESES_ORDEN[j],
                    'mes_destino': MESES_ORDEN[j + 1],
                    'calif_origen': calif_origen,
                    'calif_destino': calif_destino,
                    'peso_transicion': 1 / N_TRANSICIONES
                })

    transiciones_detalle = pd.DataFrame(detalles)
    pesos_por_id = pd.DataFrame(resumen_pesos)

    if transiciones_detalle.empty:
        return (
            transiciones_detalle,
            pd.DataFrame(),
            pd.DataFrame(),
            pesos_por_id
        )

    # Matriz ponderada
    matriz_ponderada = pd.pivot_table(
        transiciones_detalle,
        index='calif_origen',
        columns='calif_destino',
        values='peso_transicion',
        aggfunc='sum',
        fill_value=0
    )

    # Ordenar alfabeticamente las calificaciones
    filas = sorted(matriz_ponderada.index.tolist())
    cols = sorted(matriz_ponderada.columns.tolist())
    matriz_ponderada = matriz_ponderada.reindex(index=filas, columns=cols, fill_value=0)

    # Matriz de probabilidades por renglón
    suma_filas = matriz_ponderada.sum(axis=1)
    matriz_prob = matriz_ponderada.div(suma_filas.replace(0, np.nan), axis=0).fillna(0)

    return transiciones_detalle, matriz_ponderada, matriz_prob, pesos_por_id

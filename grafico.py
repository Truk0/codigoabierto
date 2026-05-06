import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def graficar_utilidad_acumulada(
    df,
    col_mes="mes",
    col_utilidad="utilidad",
    titulo="Utilidad acumulada por mes"
):
    data = df.copy()

    # ---------------------------------------------------
    # 1. Limpieza robusta de nombres de columnas
    # ---------------------------------------------------
    data.columns = data.columns.astype(str).str.strip()

    # Buscar columnas sin importar mayúsculas/minúsculas
    columnas_lookup = {c.lower(): c for c in data.columns}

    if col_mes.lower() not in columnas_lookup:
        raise KeyError(
            f"No encontré la columna '{col_mes}'. "
            f"Columnas disponibles: {list(data.columns)}"
        )

    if col_utilidad.lower() not in columnas_lookup:
        raise KeyError(
            f"No encontré la columna '{col_utilidad}'. "
            f"Columnas disponibles: {list(data.columns)}"
        )

    col_mes_real = columnas_lookup[col_mes.lower()]
    col_utilidad_real = columnas_lookup[col_utilidad.lower()]

    # ---------------------------------------------------
    # 2. Conversión robusta del mes
    # ---------------------------------------------------
    meses_es = {
        "enero": "01", "febrero": "02", "marzo": "03",
        "abril": "04", "mayo": "05", "junio": "06",
        "julio": "07", "agosto": "08", "septiembre": "09",
        "setiembre": "09", "octubre": "10", "noviembre": "11",
        "diciembre": "12"
    }

    def convertir_mes(valor):
        if pd.isna(valor):
            return pd.NaT

        texto = str(valor).strip().lower()

        # Caso: "Enero 2025"
        partes = texto.split()
        if len(partes) == 2 and partes[0] in meses_es:
            return pd.to_datetime(f"{partes[1]}-{meses_es[partes[0]]}-01")

        # Caso: fechas normales tipo "2025-01-01", "01/01/2025", etc.
        return pd.to_datetime(valor, errors="coerce", dayfirst=True)

    data["_mes_fecha"] = data[col_mes_real].apply(convertir_mes)

    if data["_mes_fecha"].isna().any():
        valores_malos = data.loc[data["_mes_fecha"].isna(), col_mes_real].unique()
        raise ValueError(
            f"Hay valores de mes que no se pudieron convertir: {valores_malos}"
        )

    data["_utilidad"] = pd.to_numeric(data[col_utilidad_real], errors="coerce").fillna(0)

    # ---------------------------------------------------
    # 3. Agrupación mensual sin pd.Grouper
    # ---------------------------------------------------
    data["_mes_periodo"] = data["_mes_fecha"].dt.to_period("M").dt.to_timestamp()

    mensual = (
        data
        .groupby("_mes_periodo", as_index=False)
        .agg(utilidad_mensual=("_utilidad", "sum"))
        .sort_values("_mes_periodo")
    )

    mensual["utilidad_acumulada"] = mensual["utilidad_mensual"].cumsum()
    mensual["mes_texto"] = mensual["_mes_periodo"].dt.strftime("%b-%Y")

    mensual["color_barra"] = np.where(
        mensual["utilidad_mensual"] >= 0,
        "#2E7D32",
        "#C62828"
    )

    # ---------------------------------------------------
    # 4. Gráfico
    # ---------------------------------------------------
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Bar(
            x=mensual["mes_texto"],
            y=mensual["utilidad_mensual"],
            name="Utilidad mensual",
            marker_color=mensual["color_barra"],
            opacity=0.75,
            hovertemplate=(
                "<b>%{x}</b><br>"
                "Utilidad mensual: $%{y:,.0f}<extra></extra>"
            )
        ),
        secondary_y=False
    )

    fig.add_trace(
        go.Scatter(
            x=mensual["mes_texto"],
            y=mensual["utilidad_acumulada"],
            name="Utilidad acumulada",
            mode="lines+markers",
            line=dict(width=4, color="#1F3A5F"),
            marker=dict(size=8, color="#1F3A5F"),
            fill="tozeroy",
            fillcolor="rgba(31, 58, 95, 0.12)",
            hovertemplate=(
                "<b>%{x}</b><br>"
                "Utilidad acumulada: $%{y:,.0f}<extra></extra>"
            )
        ),
        secondary_y=True
    )

    fig.add_hline(
        y=0,
        line_width=1.5,
        line_dash="dash",
        line_color="gray"
    )

    utilidad_final = mensual["utilidad_acumulada"].iloc[-1]
    mejor_mes = mensual.loc[mensual["utilidad_mensual"].idxmax()]
    peor_mes = mensual.loc[mensual["utilidad_mensual"].idxmin()]

    fig.add_annotation(
        x=mensual["mes_texto"].iloc[-1],
        y=utilidad_final,
        text=f"Cierre acumulado:<br><b>${utilidad_final:,.0f}</b>",
        showarrow=True,
        arrowhead=2,
        ax=40,
        ay=-50,
        bgcolor="white",
        bordercolor="#1F3A5F",
        borderwidth=1
    )

    fig.add_annotation(
        x=mejor_mes["mes_texto"],
        y=mejor_mes["utilidad_mensual"],
        text=f"Mejor mes<br><b>${mejor_mes['utilidad_mensual']:,.0f}</b>",
        showarrow=True,
        arrowhead=2,
        ax=0,
        ay=-45,
        bgcolor="white",
        bordercolor="#2E7D32",
        borderwidth=1
    )

    fig.add_annotation(
        x=peor_mes["mes_texto"],
        y=peor_mes["utilidad_mensual"],
        text=f"Peor mes<br><b>${peor_mes['utilidad_mensual']:,.0f}</b>",
        showarrow=True,
        arrowhead=2,
        ax=0,
        ay=45,
        bgcolor="white",
        bordercolor="#C62828",
        borderwidth=1
    )

    fig.update_layout(
        title=dict(
            text=titulo,
            x=0.02,
            xanchor="left",
            font=dict(size=24, family="Arial Black")
        ),
        template="plotly_white",
        height=650,
        width=1100,
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.03,
            xanchor="right",
            x=1
        ),
        margin=dict(l=70, r=70, t=100, b=80),
        plot_bgcolor="white",
        paper_bgcolor="white"
    )

    fig.update_xaxes(
        title_text="Mes",
        tickangle=-45,
        showgrid=False
    )

    fig.update_yaxes(
        title_text="Utilidad mensual",
        tickprefix="$",
        separatethousands=True,
        secondary_y=False
    )

    fig.update_yaxes(
        title_text="Utilidad acumulada",
        tickprefix="$",
        separatethousands=True,
        secondary_y=True
    )

    fig.show()

    return mensual

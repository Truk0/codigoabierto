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

    # -----------------------------
    # 1. Conversión robusta de mes
    # -----------------------------
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

        # Caso tipo: "Enero 2025"
        partes = texto.split()
        if len(partes) == 2 and partes[0] in meses_es:
            return pd.to_datetime(f"{partes[1]}-{meses_es[partes[0]]}-01")

        # Caso fecha estándar: "2025-01-01", "01/01/2025", etc.
        return pd.to_datetime(valor, errors="coerce", dayfirst=True)

    data[col_mes] = data[col_mes].apply(convertir_mes)

    if data[col_mes].isna().any():
        raise ValueError("Hay valores de mes que no se pudieron convertir a fecha.")

    # -----------------------------
    # 2. Agrupación mensual
    # -----------------------------
    mensual = (
        data
        .groupby(pd.Grouper(key=col_mes, freq="MS"), as_index=False)
        .agg(utilidad_mensual=(col_utilidad, "sum"))
        .sort_values(col_mes)
    )

    mensual["utilidad_acumulada"] = mensual["utilidad_mensual"].cumsum()
    mensual["mes_texto"] = mensual[col_mes].dt.strftime("%b-%Y")

    mensual["color_barra"] = np.where(
        mensual["utilidad_mensual"] >= 0,
        "#2E7D32",   # verde
        "#C62828"    # rojo
    )

    # -----------------------------
    # 3. Métricas para anotaciones
    # -----------------------------
    utilidad_final = mensual["utilidad_acumulada"].iloc[-1]
    mejor_mes = mensual.loc[mensual["utilidad_mensual"].idxmax()]
    peor_mes = mensual.loc[mensual["utilidad_mensual"].idxmin()]

    # -----------------------------
    # 4. Gráfico sofisticado
    # -----------------------------
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Barras de utilidad mensual
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

    # Línea de utilidad acumulada
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

    # Línea horizontal en cero
    fig.add_hline(
        y=0,
        line_width=1.5,
        line_dash="dash",
        line_color="gray"
    )

    # Anotación del cierre acumulado
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

    # Anotación mejor mes
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

    # Anotación peor mes
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

    # Diseño general
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
        barmode="relative",
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

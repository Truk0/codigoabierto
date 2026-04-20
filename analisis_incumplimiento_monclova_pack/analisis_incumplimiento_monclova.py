import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter, PercentFormatter
from pathlib import Path

INPUT_FILE = Path("ConsolidadoMonclova.xlsx")
OUTPUT_DIR = Path("analisis_monclova_output")
OUTPUT_DIR.mkdir(exist_ok=True)

def main():
    df = pd.read_excel(INPUT_FILE)
    for col in ["fecha_operacion", "fecha_vencimiento"]:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    df["month"] = df["fecha_operacion"].dt.to_period("M").dt.to_timestamp()
    df["volumen_usd_round"] = df["volumen_usd"].round(2)

    monthly = df.groupby("month").agg(
        operaciones=("nm_operacion", "count"),
        volumen_usd=("volumen_usd", "sum"),
        volumen_mxn=("volumen_mxn", "sum"),
        ticket_usd=("volumen_usd", "mean"),
        dias_activos=("fecha_operacion", lambda s: s.dt.normalize().nunique()),
    ).reset_index()

    for cls in ["SPOT", "Prepago", "Tiron"]:
        temp = df[df["clasificacion"] == cls].groupby("month").agg(
            **{f"usd_{cls.lower()}": ("volumen_usd", "sum"),
               f"ops_{cls.lower()}": ("nm_operacion", "count")}
        ).reset_index()
        monthly = monthly.merge(temp, on="month", how="left")

    monthly = monthly.fillna(0)
    monthly["share_usd_prepago"] = monthly["usd_prepago"] / monthly["volumen_usd"]
    monthly["share_usd_spot"] = monthly["usd_spot"] / monthly["volumen_usd"]
    monthly["ops_por_dia"] = monthly["operaciones"] / monthly["dias_activos"]

    monthly_ticket_patterns = []
    for month, g in df.groupby("month"):
        monthly_ticket_patterns.append({
            "month": month,
            "share_usd_top2_tickets": g.loc[g["volumen_usd_round"].isin([1000000, 1250000]), "volumen_usd"].sum() / g["volumen_usd"].sum(),
            "share_ops_top2_tickets": g["volumen_usd_round"].isin([1000000, 1250000]).mean(),
        })
    monthly_ticket_patterns = pd.DataFrame(monthly_ticket_patterns).sort_values("month")

    daily = []
    for day, g in df.groupby("fecha_operacion"):
        usd_C = g.loc[g["tipo_operacion"] == "C", "volumen_usd"].sum()
        usd_V = g.loc[g["tipo_operacion"] == "V", "volumen_usd"].sum()
        daily.append({
            "fecha_operacion": day,
            "month": pd.Timestamp(day).to_period("M").to_timestamp(),
            "ops": len(g),
            "usd_total": g["volumen_usd"].sum(),
            "both_C_V": int((g["tipo_operacion"] == "C").any() and (g["tipo_operacion"] == "V").any()),
            "net_C_minus_V": usd_C - usd_V,
        })
    daily = pd.DataFrame(daily).sort_values("fecha_operacion")
    month_day_stats = daily.groupby("month").agg(
        pct_days_multiop=("ops", lambda s: (s > 1).mean())
    ).reset_index()

    monthly_dir = df.groupby(["month", "tipo_operacion"]).agg(usd=("volumen_usd", "sum")).reset_index()
    monthly_dir_p = monthly_dir.pivot(index="month", columns="tipo_operacion", values="usd").fillna(0)
    monthly_dir_p["gross_CV"] = monthly_dir_p.get("C", 0) + monthly_dir_p.get("V", 0)
    monthly_dir_p["net_C_minus_V"] = monthly_dir_p.get("C", 0) - monthly_dir_p.get("V", 0)
    monthly_dir_p["turnover_to_net_ratio"] = monthly_dir_p["gross_CV"] / monthly_dir_p["net_C_minus_V"].abs().replace(0, np.nan)
    monthly_dir_p = monthly_dir_p.reset_index()

    monthly.to_csv(OUTPUT_DIR / "monthly_features.csv", index=False)
    monthly_ticket_patterns.to_csv(OUTPUT_DIR / "monthly_ticket_patterns.csv", index=False)
    daily.to_csv(OUTPUT_DIR / "daily_metrics.csv", index=False)
    month_day_stats.to_csv(OUTPUT_DIR / "month_day_stats.csv", index=False)
    monthly_dir_p.to_csv(OUTPUT_DIR / "monthly_directionality.csv", index=False)

    usd_fmt = FuncFormatter(lambda x, p: f"${x/1e6:,.1f}M")

    fig, ax1 = plt.subplots(figsize=(11, 5.5))
    ax1.bar(monthly["month"], monthly["volumen_usd"], width=23)
    ax1.set_title("Histórico mensual de volumen USD y número de operaciones")
    ax1.set_ylabel("Volumen USD")
    ax1.yaxis.set_major_formatter(usd_fmt)
    ax2 = ax1.twinx()
    ax2.plot(monthly["month"], monthly["operaciones"], marker="o")
    ax2.set_ylabel("Operaciones")
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "01_historial_volumen_operaciones.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    pivot_cls = df.pivot_table(index="month", columns="clasificacion", values="volumen_usd", aggfunc="sum", fill_value=0)
    fig, ax = plt.subplots(figsize=(11, 5.5))
    bottom = np.zeros(len(pivot_cls))
    for col in ["SPOT", "Prepago", "Tiron"]:
        vals = pivot_cls[col] if col in pivot_cls.columns else np.zeros(len(pivot_cls))
        ax.bar(pivot_cls.index, vals, bottom=bottom, width=23, label=col)
        bottom = bottom + np.array(vals)
    ax.set_title("Mix mensual por producto (volumen USD)")
    ax.set_ylabel("Volumen USD")
    ax.yaxis.set_major_formatter(usd_fmt)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.xticks(rotation=45, ha="right")
    ax.legend()
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "02_mix_producto_usd.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.plot(monthly["month"], monthly["share_usd_prepago"], marker="o", label="% USD Prepago")
    ax.plot(monthly_ticket_patterns["month"], monthly_ticket_patterns["share_usd_top2_tickets"], marker="o", label="% USD tickets 1M/1.25M")
    ax.plot(monthly["month"], monthly["operaciones"] / monthly["operaciones"].max(), marker="o", label="Operaciones (índice)")
    ax.set_title("Señales de presión operativa / liquidez")
    ax.set_ylabel("Proporción / Índice")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.xticks(rotation=45, ha="right")
    ax.legend()
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "03_indicadores_liquidez.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.bar(monthly_dir_p["month"], monthly_dir_p["gross_CV"], width=23, label="Gross C+V")
    ax.plot(monthly_dir_p["month"], monthly_dir_p["net_C_minus_V"].abs(), marker="o", label="|Neto C-V|")
    ax.set_title("Gross operado vs. desbalance neto mensual (tipos C y V)")
    ax.set_ylabel("USD")
    ax.yaxis.set_major_formatter(usd_fmt)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.xticks(rotation=45, ha="right")
    ax.legend()
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "04_gross_vs_neto.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    jan = df[df["month"] == pd.Timestamp("2026-01-01")].copy()
    pivot_jan = jan.pivot_table(index="fecha_operacion", columns="clasificacion", values="volumen_usd", aggfunc="sum", fill_value=0)
    fig, ax = plt.subplots(figsize=(11, 5.5))
    bottom = np.zeros(len(pivot_jan))
    for col in ["SPOT", "Prepago", "Tiron"]:
        vals = pivot_jan[col] if col in pivot_jan.columns else np.zeros(len(pivot_jan))
        ax.bar(pivot_jan.index, vals, bottom=bottom, width=0.8, label=col)
        bottom = bottom + np.array(vals)
    ax.set_title("Enero 2026: volumen diario por producto")
    ax.set_ylabel("Volumen USD")
    ax.yaxis.set_major_formatter(usd_fmt)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    plt.xticks(rotation=45, ha="right")
    ax.legend()
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "05_enero2026_diario_producto.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.plot(monthly_ticket_patterns["month"], monthly_ticket_patterns["share_usd_top2_tickets"], marker="o", label="% USD tickets 1M/1.25M")
    ax.plot(monthly_ticket_patterns["month"], monthly_ticket_patterns["share_ops_top2_tickets"], marker="o", label="% ops tickets 1M/1.25M")
    ax.set_title("Concentración mensual en tickets estandarizados")
    ax.set_ylabel("Proporción")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.xticks(rotation=45, ha="right")
    ax.legend()
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "06_concentracion_tickets.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    print("Análisis generado en:", OUTPUT_DIR.resolve())

if __name__ == "__main__":
    main()

import os, json, pandas as pd, numpy as np, matplotlib.pyplot as plt
from artifact_tool import Blob, SpreadsheetFile

outdir = "/mnt/data/analisis_efactor_nuevo_output"
os.makedirs(outdir, exist_ok=True)

wb = SpreadsheetFile.import_xlsx(Blob.load("/mnt/data/e_factor.xlsx"))
sheet = wb.sheets.get_active()
data = sheet.get_used_range().values
headers = data[0]
rows = data[1:]
df = pd.DataFrame(rows, columns=headers)

for col in ["fecha_operacion","fecha_confirmacion"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")
df["fecha_operacion_dt"] = pd.to_datetime(df["fecha_operacion"], unit="D", origin="1899-12-30", errors="coerce")
df["fecha_liquidacion_dt"] = pd.to_datetime(df["fecha_liquidacion"], errors="coerce")
df["hora_operacion"] = df["hora_operacion"].astype(str)
df["hora_operacion_dt"] = pd.to_datetime(df["hora_operacion"], format="%H:%M:%S", errors="coerce")
df["hour"] = df["hora_operacion_dt"].dt.hour
df["month"] = df["fecha_operacion_dt"].dt.to_period("M").astype(str)

fx = df[df["tipo_operacion"].isin(["C","V"])].copy()
fx["is_madrugada"] = fx["hour"].between(0,5)

hour_stats = fx.groupby("hour").agg(
    operaciones=("nm_operacion","count"),
    volumen_mxn=("volumen_mxn","sum"),
    dias_activos=("fecha_operacion_dt","nunique")
).reset_index()
hour_stats["pct_operaciones"] = hour_stats["operaciones"]/hour_stats["operaciones"].sum()
hour_stats["pct_volumen"] = hour_stats["volumen_mxn"]/hour_stats["volumen_mxn"].sum()

monthly = fx.groupby("month").agg(ops_total=("nm_operacion","count"), vol_mxn_total=("volumen_mxn","sum")).reset_index()
madr = fx[fx["is_madrugada"]].groupby("month").agg(ops_madr=("nm_operacion","count"), vol_mxn_madr=("volumen_mxn","sum")).reset_index()
monthly_h = monthly.merge(madr, on="month", how="left").fillna(0)
monthly_h["pct_ops_madr"] = monthly_h["ops_madr"]/monthly_h["ops_total"]
monthly_h["pct_vol_madr"] = monthly_h["vol_mxn_madr"]/monthly_h["vol_mxn_total"]

pair_records = []
for key, grp in fx.groupby(["fecha_operacion_dt","hora_operacion","volumen_usd"]):
    C0 = grp[(grp["tipo_operacion"]=="C") & (grp["plazo"]==0)].copy()
    Cf = grp[(grp["tipo_operacion"]=="C") & (grp["plazo"]>=1)].copy()
    V0 = grp[(grp["tipo_operacion"]=="V") & (grp["plazo"]==0)].copy()
    Vf = grp[(grp["tipo_operacion"]=="V") & (grp["plazo"]>=1)].copy()
    n = min(len(Cf), len(V0))
    if n:
        for i in range(n):
            a,b = Cf.iloc[i], V0.iloc[i]
            pair_records.append({
                "fecha_operacion_dt": key[0],
                "hora_operacion": key[1],
                "hour": int(pd.to_datetime(key[1], format="%H:%M:%S").hour),
                "volumen_usd": float(key[2]),
                "bridge_type":"Compra a plazo y venta mismo día",
                "future_plazo": int(a["plazo"]),
                "future_liq": a["fecha_liquidacion_dt"],
                "mxn_notional": float((a["volumen_mxn"] + b["volumen_mxn"])/2),
                "sign": -1
            })
    n = min(len(Vf), len(C0))
    if n:
        for i in range(n):
            a,b = Vf.iloc[i], C0.iloc[i]
            pair_records.append({
                "fecha_operacion_dt": key[0],
                "hora_operacion": key[1],
                "hour": int(pd.to_datetime(key[1], format="%H:%M:%S").hour),
                "volumen_usd": float(key[2]),
                "bridge_type":"Compra hoy y venta a plazo",
                "future_plazo": int(a["plazo"]),
                "future_liq": a["fecha_liquidacion_dt"],
                "mxn_notional": float((a["volumen_mxn"] + b["volumen_mxn"])/2),
                "sign": 1
            })
pairs = pd.DataFrame(pair_records)

daily = fx.assign(plazo_grp=np.where(fx["plazo"]==0,"0","fut")).groupby(["fecha_operacion_dt","tipo_operacion","plazo_grp"])["volumen_usd"].sum().unstack(["tipo_operacion","plazo_grp"], fill_value=0)
daily.columns = [f"{a}_{b}" for a,b in daily.columns]
daily = daily.reset_index()
for c in ["C_0","C_fut","V_0","V_fut"]:
    if c not in daily.columns:
        daily[c] = 0.0
daily["paired_Cfut_V0_usd"] = np.minimum(daily["C_fut"], daily["V_0"])
daily["paired_Vfut_C0_usd"] = np.minimum(daily["V_fut"], daily["C_0"])

calendar = pd.date_range(pairs["fecha_operacion_dt"].min(), pairs["future_liq"].max(), freq="D")
pos_records = []
for d in calendar:
    active = pairs[(pairs["fecha_operacion_dt"] <= d) & (pairs["future_liq"] > d)]
    pos_records.append({
        "date": d,
        "net_usd": float((active["sign"] * active["volumen_usd"]).sum()),
        "gross_usd": float(active["volumen_usd"].sum()),
        "net_mxn": float((active["sign"] * active["mxn_notional"]).sum()),
        "gross_mxn": float(active["mxn_notional"].sum()),
        "pairs_active": int(len(active))
    })
pos = pd.DataFrame(pos_records)
trade_dates = pd.to_datetime(sorted(fx["fecha_operacion_dt"].dropna().unique()))
pos_trade = pos[pos["date"].isin(trade_dates)].copy()
limit = 30_000_000

summary = {
    "ops_total_cv": int(len(fx)),
    "dias_operados": int(fx["fecha_operacion_dt"].nunique()),
    "pct_ops_hora_14": float(hour_stats.loc[hour_stats["hour"]==14,"pct_operaciones"].sum()),
    "pct_ops_horas_13_15": float(hour_stats.loc[hour_stats["hour"].between(13,15),"pct_operaciones"].sum()),
    "pct_ops_madrugada_00_05": float(fx["is_madrugada"].mean()),
    "pct_vol_madrugada_00_05": float(fx.loc[fx["is_madrugada"],"volumen_mxn"].sum()/fx["volumen_mxn"].sum()),
    "casos_exactos_puente": int(len(pairs)),
    "casos_compra_plazo_venta_hoy": int(pairs["bridge_type"].str.contains("plazo y venta mismo día").sum()),
    "casos_compra_hoy_venta_plazo": int(pairs["bridge_type"].str.contains("Compra hoy y venta a plazo").sum()),
    "trade_days_net_abs_gt_30m": int((pos_trade["net_mxn"].abs() > limit).sum()),
    "trade_days_gross_gt_30m": int((pos_trade["gross_mxn"] > limit).sum()),
    "max_net_abs_trade_mxn": float(pos_trade["net_mxn"].abs().max()),
    "max_gross_trade_mxn": float(pos_trade["gross_mxn"].max())
}

with open(f"{outdir}/summary_metrics.json","w",encoding="utf-8") as f:
    json.dump(summary,f,ensure_ascii=False,indent=2)

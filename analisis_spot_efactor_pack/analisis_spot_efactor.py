
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import os

INPUT_FILE = "Efactor.xlsx"
OUTDIR = "analisis_efactor_output"

def main():
    os.makedirs(OUTDIR, exist_ok=True)

    df = pd.read_excel(INPUT_FILE)
    df["fecha_operacion"] = pd.to_datetime(df["fecha_operacion"])
    df = df[df["clasificacion"].eq("SPOT")].copy()
    ops = df[df["tipo_operacion"].isin(["C", "V"])].copy()
    ops = ops.sort_values(["fecha_operacion", "nm_operacion"]).reset_index(drop=True)

    # Tipo de cambio implícito y lado cliente
    ops["rate"] = ops["volumen_mxn"] / ops["volumen_usd"]
    ops["client_side"] = np.where(ops["tipo_operacion"] == "V", "BUY_USD", "SELL_USD")
    ops["client_qty_signed"] = np.where(ops["client_side"] == "BUY_USD", ops["volumen_usd"], -ops["volumen_usd"])
    ops["month"] = ops["fecha_operacion"].dt.to_period("M").astype(str)
    ops["inventory_usd"] = ops["client_qty_signed"].cumsum()
    ops["vol_round"] = ops["volumen_usd"].round(2)

    daily = ops.groupby("fecha_operacion").agg(
        ops=("nm_operacion", "count"),
        buy_ops=("client_side", lambda s: (s == "BUY_USD").sum()),
        sell_ops=("client_side", lambda s: (s == "SELL_USD").sum()),
        client_net_usd=("client_qty_signed", "sum"),
        usd_gross=("volumen_usd", "sum"),
        avg_rate=("rate", lambda s: np.average(s, weights=ops.loc[s.index, "volumen_usd"])),
        inv_close=("inventory_usd", "last"),
    )
    daily["same_day_two_way"] = (daily["buy_ops"] > 0) & (daily["sell_ops"] > 0)
    daily["net_to_gross"] = daily["client_net_usd"].abs() / daily["usd_gross"]
    daily["month"] = daily.index.to_period("M").astype(str)
    daily["next_rate"] = daily["avg_rate"].shift(-1)
    daily["next_change"] = daily["next_rate"] - daily["avg_rate"]

    monthly = ops.groupby("month").agg(
        ops=("nm_operacion", "count"),
        active_days=("fecha_operacion", "nunique"),
        usd_gross=("volumen_usd", "sum"),
        client_net_usd=("client_qty_signed", "sum"),
        avg_rate=("rate", lambda s: np.average(s, weights=ops.loc[s.index, "volumen_usd"])),
    )
    monthly["abs_net_usd"] = monthly["client_net_usd"].abs()
    monthly["net_to_gross"] = monthly["abs_net_usd"] / monthly["usd_gross"]

    # Empate FIFO
    longs, shorts = deque(), deque()
    matches = []

    for _, row in ops.iterrows():
        date = row["fecha_operacion"]
        opid = row["nm_operacion"]
        qty = row["client_qty_signed"]
        rate = row["rate"]

        if qty > 0:
            remaining = qty
            while remaining > 1e-9 and shorts:
                sh = shorts[0]
                close_qty = min(remaining, sh["remaining"])
                pnl = (sh["open_rate"] - rate) * close_qty
                matches.append({
                    "open_date": sh["open_date"],
                    "close_date": date,
                    "hold_days": (date - sh["open_date"]).days,
                    "direction": "SHORT_USD",
                    "qty_usd": close_qty,
                    "open_rate": sh["open_rate"],
                    "close_rate": rate,
                    "pnl_mxn": pnl,
                    "open_opid": sh["open_opid"],
                    "close_opid": opid,
                })
                sh["remaining"] -= close_qty
                remaining -= close_qty
                if sh["remaining"] <= 1e-9:
                    shorts.popleft()
            if remaining > 1e-9:
                longs.append({"open_date": date, "open_rate": rate, "remaining": remaining, "open_opid": opid})
        else:
            remaining = -qty
            while remaining > 1e-9 and longs:
                lg = longs[0]
                close_qty = min(remaining, lg["remaining"])
                pnl = (rate - lg["open_rate"]) * close_qty
                matches.append({
                    "open_date": lg["open_date"],
                    "close_date": date,
                    "hold_days": (date - lg["open_date"]).days,
                    "direction": "LONG_USD",
                    "qty_usd": close_qty,
                    "open_rate": lg["open_rate"],
                    "close_rate": rate,
                    "pnl_mxn": pnl,
                    "open_opid": lg["open_opid"],
                    "close_opid": opid,
                })
                lg["remaining"] -= close_qty
                remaining -= close_qty
                if lg["remaining"] <= 1e-9:
                    longs.popleft()
            if remaining > 1e-9:
                shorts.append({"open_date": date, "open_rate": rate, "remaining": remaining, "open_opid": opid})

    matches_df = pd.DataFrame(matches)
    matches_df["close_month"] = matches_df["close_date"].dt.to_period("M").astype(str)

    # Exportar resultados
    daily.to_csv(os.path.join(OUTDIR, "daily_summary.csv"))
    monthly.to_csv(os.path.join(OUTDIR, "monthly_summary.csv"))
    matches_df.to_csv(os.path.join(OUTDIR, "roundtrips_fifo.csv"), index=False)

    # Gráfico simple
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(monthly.index, monthly["usd_gross"] / 1e6, label="Bruto USD ($mm)")
    ax.plot(monthly.index, monthly["abs_net_usd"] / 1e6, marker="o", label="Neto abs USD ($mm)")
    ax.tick_params(axis="x", rotation=60)
    ax.legend()
    ax.set_title("Volumen mensual spot")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "volumen_mensual.png"), dpi=160)

    # Resumen en consola
    hit_rate = (
        (np.sign(daily.loc[daily["client_net_usd"] != 0, "client_net_usd"]) *
         np.sign(daily.loc[daily["client_net_usd"] != 0, "next_change"])) > 0
    ).mean()

    print("Operaciones analizadas:", len(ops))
    print("Dias activos:", ops["fecha_operacion"].nunique())
    print("Volumen bruto USD:", round(ops["volumen_usd"].sum(), 2))
    print("Neto final / bruto:", round(abs(ops["client_qty_signed"].sum()) / ops["volumen_usd"].sum(), 6))
    print("% dias compra+venta mismo dia:", round(daily["same_day_two_way"].mean(), 4))
    print("% volumen round-trip cerrado mismo dia:",
          round(matches_df.loc[matches_df["hold_days"] == 0, "qty_usd"].sum() / matches_df["qty_usd"].sum(), 4))
    print("% volumen round-trip <=1 dia:",
          round(matches_df.loc[matches_df["hold_days"] <= 1, "qty_usd"].sum() / matches_df["qty_usd"].sum(), 4))
    print("P&L implicito total MXN:", round(matches_df["pnl_mxn"].sum(), 2))
    print("Hit rate direccional siguiente dia:", round(hit_rate, 4))

if __name__ == "__main__":
    main()

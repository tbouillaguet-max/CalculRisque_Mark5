import sys, os
import pandas as pd

run = sys.argv[1]
pd.set_option("display.width", 200)
pd.set_option("display.float_format", "{:,.0f}".format)

f = pd.read_parquet(os.path.join(run, "executions.parquet"))
f["friction"] = f["commission"] + f["slippage"]

print("=== FRICTION PAR MOTIF DE FILL ===")
g = f.groupby("reason")[["commission", "slippage", "friction"]].sum()
g["n_fills"] = f.groupby("reason").size()
g["friction_par_fill"] = g["friction"] / g["n_fills"]
g["pct_friction"] = 100 * g["friction"] / g["friction"].sum()
print(g.sort_values("friction", ascending=False))

print()
print("=== FILLS PAR MOTIF ET SENS ===")
print(f.groupby(["reason", "side"]).size())

print()
print("TOTAL friction  :", round(f["friction"].sum()))
print("dont commissions:", round(f["commission"].sum()))
print("dont slippage   :", round(f["slippage"].sum()))
print("nombre de fills :", len(f))

t = pd.read_parquet(os.path.join(run, "trades.parquet"))
if "entry_date" in t.columns and "exit_date" in t.columns:
    t["jours"] = (pd.to_datetime(t["exit_date"]) - pd.to_datetime(t["entry_date"])).dt.days
    print()
    print("=== DUREE DE DETENTION (jours) PAR MOTIF DE SORTIE ===")
    print(t.groupby("exit_reason")["jours"].describe()[["count", "mean", "50%", "min", "max"]])

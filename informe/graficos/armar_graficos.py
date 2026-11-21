"""Gráficos de los modelos para el informe (sección 4).

Usa las mismas funciones y particiones que etapa2/entrenar_modelos.py: los modelos se vuelven a ajustar
solo con la parte de entrenamiento, así las curvas y las series son las del test. Correr desde esta carpeta:
    python armar_graficos.py
Necesita matplotlib, además de lo de requirements.txt.
"""

import json
import sys
from pathlib import Path

import matplotlib
import matplotlib.dates

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split

AQUI = Path(__file__).parent
ETAPA2 = AQUI.parents[1] / "etapa2"
sys.path.insert(0, str(ETAPA2))
from entrenar_modelos import SEMILLA, modelo_lineal_ventas, pipeline_lightgbm_clf, pipeline_logistica  # noqa: E402
from preparacion import CAT_ATTRITION, COLUMNAS_VENTAS, NUM_ATTRITION, objetivo_attrition  # noqa: E402

AZUL, NARANJA = "#2a78d6", "#eb6834"             # paleta validada (modo claro)
TINTA, TINTA_2, GRILLA = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.edgecolor": GRILLA, "axes.labelcolor": TINTA_2,
                     "xtick.color": TINTA_2, "ytick.color": TINTA_2, "axes.titlesize": 9, "axes.titlecolor": TINTA,
                     "axes.titleweight": "bold", "axes.titlelocation": "left", "legend.frameon": False,
                     "axes.spines.top": False, "axes.spines.right": False})
COMA = FuncFormatter(lambda v, _: f"{v:,.1f}".replace(",", " ").replace(".", ",").replace(" ", ".").rstrip("0").rstrip(","))
MILES = FuncFormatter(lambda v, _: f"{v:,.0f}".replace(",", "."))


def coma(x: float, decimales: int = 3) -> str:
    return f"{x:.{decimales}f}".replace(".", ",")


def figura_attrition():
    empleados = pd.read_csv(ETAPA2 / "datos" / "etapa2" / "empleados.csv")
    X, y = empleados[CAT_ATTRITION + NUM_ATTRITION], objetivo_attrition(empleados)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=SEMILLA, stratify=y)
    metricas = json.loads((ETAPA2 / "modelos" / "metricas.json").read_text(encoding="utf-8"))["attrition"]

    fig, (izq, der) = plt.subplots(1, 2, figsize=(7.2, 2.9), gridspec_kw={"width_ratios": [1, 1.25], "wspace": 0.55})
    izq.plot([0, 1], [0, 1], color=GRILLA, lw=1, ls="--", zorder=1)
    for nombre, clave, fabrica, color in [("Logística", "logistica", pipeline_logistica, AZUL),
                                          ("LightGBM", "lightgbm", pipeline_lightgbm_clf, NARANJA)]:
        prob = fabrica().fit(X_tr, y_tr).predict_proba(X_te)[:, 1]
        fpr, tpr, umbrales = roc_curve(y_te, prob)
        auc = metricas["comparacion"][clave]["auc_test"]            # el mismo valor que la tabla del informe
        assert abs(auc - roc_auc_score(y_te, prob)) < 1e-3
        izq.plot(fpr, tpr, color=color, lw=2, label=f"{nombre} (AUC {coma(auc)})", zorder=2)
        if color == AZUL:
            i = int(np.argmin(np.abs(umbrales - 0.5)))
            izq.scatter(fpr[i], tpr[i], s=36, color=AZUL, edgecolor="white", linewidth=2, zorder=3)
            izq.annotate("umbral 0,5", (fpr[i], tpr[i]), xytext=(10, -12), textcoords="offset points", color=TINTA_2, fontsize=7.5)
    izq.set(xlim=(0, 1), ylim=(0, 1.02), xlabel="Falsos positivos (proporción de los que no renuncian)",
            ylabel="Verdaderos positivos (recall)")
    izq.xaxis.set_major_formatter(COMA)
    izq.yaxis.set_major_formatter(COMA)
    izq.set_title("Curva ROC en test (400 empleados)")
    izq.legend(loc="lower right", fontsize=7.5, handlelength=1.4, borderaxespad=0.2)
    izq.grid(color=GRILLA, lw=0.6)
    izq.set_axisbelow(True)

    efectos = {"sueldo_vs_mercado_pct": ("Sueldo vs. mercado", "−4,5 % por punto"),
               "satisfaccion_clima": ("Satisfacción con el clima", "−53 % por punto"),
               "meses_sin_aumento": ("Meses sin aumento", "+5,2 % por mes"),
               "horas_extra_mes": ("Horas extra", "+5,2 % por hora"),
               "antiguedad_anios": ("Antigüedad", "−8,3 % por año"),
               "capacitaciones_ultimo_anio": ("Capacitaciones", "−21 % por capacitación"),
               "distancia_km": ("Distancia", "+2,3 % por km"),
               "area": ("Área", "Atención al cliente, la de más riesgo")}
    importancia = metricas["importancia_permutacion_auc"]
    nombres = list(efectos)[::-1]
    valores = [importancia[n] for n in nombres]
    der.barh(range(len(nombres)), valores, height=0.62, color=AZUL, zorder=2)
    for k, (n, v) in enumerate(zip(nombres, valores)):
        der.text(v + 0.0012, k, efectos[n][1], va="center", fontsize=7.5, color=TINTA_2)
    der.set_yticks(range(len(nombres)), [efectos[n][0] for n in nombres], color=TINTA)
    der.tick_params(axis="y", length=0)
    der.set_xlim(0, 0.105)
    der.set_xticks([0, 0.02, 0.04, 0.06])
    der.xaxis.set_major_formatter(FuncFormatter(lambda v, _: coma(v, 2)))
    der.set_xlabel("Caída del AUC al desordenar la variable")
    der.set_title("Importancia y sentido de cada variable")
    der.grid(axis="x", color=GRILLA, lw=0.6)
    der.set_axisbelow(True)
    der.spines["left"].set_visible(False)
    fig.text(0.5, -0.06, "Barras: caída del AUC en test al desordenar la variable. Texto: efecto de una unidad más "
             "sobre las chances de renunciar (odds ratio).", ha="center", fontsize=7, color=TINTA_2)
    fig.savefig(AQUI / "fig3_attrition.png", dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def figura_ventas():
    ventas = pd.read_csv(ETAPA2 / "datos" / "etapa2" / "ventas.csv", parse_dates=["semana"])
    semanas = np.sort(ventas["semana"].unique())
    corte = semanas[int(len(semanas) * 0.8)]
    tr, te = ventas["semana"] < corte, ventas["semana"] >= corte
    modelo = modelo_lineal_ventas().fit(ventas.loc[tr, COLUMNAS_VENTAS], ventas.loc[tr, "unidades_vendidas"])
    test = ventas[te].assign(pred=modelo.predict(ventas.loc[te, COLUMNAS_VENTAS]) * 1.012)   # factor de Duan

    fig, ejes = plt.subplots(1, 2, figsize=(7.2, 2.6), sharex=True, gridspec_kw={"wspace": 0.28})
    for eje, categoria, nota in [(ejes[0], "Aires acondicionados", "sube con la temperatura"),
                                 (ejes[1], "Electro", "baja con el precio del cobre, no con el clima")]:
        serie = test[(test["categoria"] == categoria) & (test["region"] == "AMBA")].sort_values("semana")
        eje.plot(serie["semana"], serie["unidades_vendidas"], color=AZUL, lw=2, label="Real")
        eje.plot(serie["semana"], serie["pred"], color=NARANJA, lw=2, label="Modelo")
        mape = (serie["unidades_vendidas"] - serie["pred"]).abs().div(serie["unidades_vendidas"]).mean() * 100
        eje.set_title(f"{categoria}, AMBA")
        eje.text(0.01, 0.98, f"{nota[0].upper() + nota[1:]}\nError medio en test: {coma(mape, 1)} %", transform=eje.transAxes,
                 va="top", fontsize=7.5, color=TINTA_2, linespacing=1.4)
        eje.set_ylim(0, serie[["unidades_vendidas", "pred"]].to_numpy().max() * 1.28)
        eje.yaxis.set_major_formatter(MILES)
        eje.grid(axis="y", color=GRILLA, lw=0.6)
        eje.set_axisbelow(True)
        eje.tick_params(axis="x", labelsize=7)
    ejes[0].set_ylabel("Unidades por semana")
    ejes[1].legend(loc="lower right", fontsize=7.5, ncol=2, handlelength=1.4)
    meses = ["", "ene", "feb", "mar", "abr", "may", "jun", "jul", "ago", "sep", "oct", "nov", "dic"]
    for eje in ejes:
        eje.xaxis.set_major_locator(matplotlib.dates.MonthLocator(bymonth=[2, 4, 6, 8, 10]))
        eje.xaxis.set_major_formatter(FuncFormatter(lambda v, _: (lambda d: f"{meses[d.month]} {d.year % 100}")(matplotlib.dates.num2date(v))))
    fig.text(0.5, -0.03, f"Test: las últimas 40 semanas ({pd.Timestamp(corte):%d/%m/%Y} a {test['semana'].max():%d/%m/%Y}); "
             "el modelo se entrenó con las 160 anteriores.", ha="center", fontsize=7, color=TINTA_2)
    fig.savefig(AQUI / "fig4_ventas.png", dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    figura_attrition()
    figura_ventas()
    print("listo: fig3_attrition.png y fig4_ventas.png")

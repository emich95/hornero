"""
entrenar_modelos.py: entrena, valida y guarda los dos modelos de la etapa 2.

    python entrenar_modelos.py

Genera:
  - modelos/attrition.joblib  regresión logística: probabilidad de renuncia de un empleado
  - modelos/ventas.joblib     regresión lineal sobre log(unidades): ventas semanales por categoría y región
  - modelos/metricas.json     validación de cada modelo y de las alternativas descartadas

Las tools (tools.py) solo cargan estos archivos: no se reentrena en cada consulta.
"""

import json
import warnings
from datetime import datetime
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (confusion_matrix, f1_score, mean_absolute_error, mean_squared_error,
                             precision_score, r2_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, PolynomialFeatures, StandardScaler

from preparacion import (CAT_ATTRITION, CATEGORIAS, COLUMNAS_VENTAS, NUM_ATTRITION, REGIONES,
                         VARIABLES_VENTAS, diseno_ventas, nombres_diseno_ventas, objetivo_attrition)

warnings.filterwarnings("ignore")

RUTA_DATOS = Path(__file__).parent / "datos" / "etapa2"
RUTA_MODELOS = Path(__file__).parent / "modelos"
SEMILLA = 42
UMBRAL_RIESGO = 0.5          # decisión D6: umbral por defecto de "en riesgo"


def redondear(d: dict, n: int = 4) -> dict:
    return {k: (round(float(v), n) if isinstance(v, (float, np.floating)) else v) for k, v in d.items()}


# ==========================================================================
# 1. Attrition
# ==========================================================================

def pipeline_logistica() -> Pipeline:
    pre = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), CAT_ATTRITION),
                             ("num", StandardScaler(), NUM_ATTRITION)])
    return Pipeline([("pre", pre), ("modelo", LogisticRegression(max_iter=2000))])


def pipeline_lightgbm_clf() -> Pipeline:
    pre = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), CAT_ATTRITION),
                             ("num", "passthrough", NUM_ATTRITION)])
    modelo = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15,
                                random_state=SEMILLA, verbose=-1)
    return Pipeline([("pre", pre), ("modelo", modelo)])


def metricas_umbral(y_real, prob, umbral: float) -> dict:
    pred = (prob >= umbral).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_real, pred).ravel()
    return redondear({"umbral": umbral, "precision": precision_score(y_real, pred),
                      "recall": recall_score(y_real, pred), "f1": f1_score(y_real, pred),
                      "pct_marcados": pred.mean() * 100, "vp": int(tp), "fp": int(fp), "fn": int(fn), "vn": int(tn)})


def entrenar_attrition() -> dict:
    empleados = pd.read_csv(RUTA_DATOS / "empleados.csv")
    X = empleados[CAT_ATTRITION + NUM_ATTRITION]
    y = objetivo_attrition(empleados)
    activos = (y == 0).to_numpy()
    print(f"[attrition] {len(empleados)} empleados, {y.mean():.1%} renunció, {activos.sum()} activos")

    # Validación: train/test estratificado 80/20 + validación cruzada de 5 folds
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=SEMILLA, stratify=y)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEMILLA)
    comparacion = {}
    for nombre, fabrica in [("logistica", pipeline_logistica), ("lightgbm", pipeline_lightgbm_clf)]:
        auc_cv = cross_val_score(fabrica(), X, y, cv=cv, scoring="roc_auc")
        modelo = fabrica().fit(X_tr, y_tr)
        prob = modelo.predict_proba(X_te)[:, 1]
        umbrales = np.round(np.arange(0.15, 0.71, 0.01), 2)
        umbral_f1 = float(umbrales[int(np.argmax([f1_score(y_te, prob >= u) for u in umbrales]))])
        comparacion[nombre] = {
            "auc_cv_media": round(float(auc_cv.mean()), 4), "auc_cv_desvio": round(float(auc_cv.std()), 4),
            "auc_test": round(float(roc_auc_score(y_te, prob)), 4),
            "umbral_0_5": metricas_umbral(y_te, prob, UMBRAL_RIESGO),
            "umbral_f1_max": metricas_umbral(y_te, prob, umbral_f1),
        }
        if nombre == "logistica":
            modelo_test = modelo
        print(f"[attrition] {nombre:9s} AUC CV {auc_cv.mean():.3f} ± {auc_cv.std():.3f} | AUC test "
              f"{comparacion[nombre]['auc_test']:.3f} | F1 con 0,5: {comparacion[nombre]['umbral_0_5']['f1']:.2f}")

    # Interpretación 1: cuánto pesa cada variable (caída del AUC en test al desordenarla)
    perm = permutation_importance(modelo_test, X_te, y_te, scoring="roc_auc", n_repeats=20, random_state=SEMILLA)
    importancia = {col: round(float(m), 4) for col, m in zip(X.columns, perm.importances_mean)}

    # Modelo final: se entrena con los 2.000 empleados
    final = pipeline_logistica().fit(X, y)
    pre, logit = final.named_steps["pre"], final.named_steps["modelo"]
    coef = dict(zip(pre.get_feature_names_out(), logit.coef_[0]))
    escalador = pre.named_transformers_["num"]

    # Interpretación 2: en qué sentido actúa cada variable
    efectos_num = {}
    for i, col in enumerate(NUM_ATTRITION):
        b = coef[f"num__{col}"]
        efectos_num[col] = redondear({"coef_estandarizado": b, "desvio": escalador.scale_[i],
                                      "odds_ratio_por_unidad": np.exp(b / escalador.scale_[i]),
                                      "odds_ratio_por_desvio": np.exp(b)})
    efectos_cat = {col: {k.split("__", 1)[1][len(col) + 1:]: round(float(v), 4) for k, v in coef.items()
                         if k.startswith(f"cat__{col}_")} for col in CAT_ATTRITION}

    # Chequeo de la decisión D7: el riesgo se calcula sobre activos con el modelo entrenado con todos.
    # Comparamos contra predicciones fuera de muestra para medir cuánto infla (o desinfla) eso.
    prob_dentro = final.predict_proba(X)[:, 1]
    prob_fuera = cross_val_predict(pipeline_logistica(), X, y, cv=cv, method="predict_proba")[:, 1]
    chequeo_activos = {
        "en_riesgo_activos_dentro_de_muestra": int((prob_dentro[activos] >= UMBRAL_RIESGO).sum()),
        "en_riesgo_activos_fuera_de_muestra": int((prob_fuera[activos] >= UMBRAL_RIESGO).sum()),
        "prob_media_activos_dentro": round(float(prob_dentro[activos].mean()), 4),
        "prob_media_activos_fuera": round(float(prob_fuera[activos].mean()), 4),
    }
    print(f"[attrition] activos en riesgo (u=0,5): {chequeo_activos['en_riesgo_activos_dentro_de_muestra']} "
          f"dentro de muestra vs {chequeo_activos['en_riesgo_activos_fuera_de_muestra']} fuera de muestra")

    artefacto = {
        "pipeline": final,
        "features_cat": CAT_ATTRITION,
        "features_num": NUM_ATTRITION,
        "valores_cat": {c: sorted(empleados[c].unique().tolist()) for c in CAT_ATTRITION},
        "rangos_num": {c: [float(empleados[c].min()), float(empleados[c].max())] for c in NUM_ATTRITION},
        "umbral": UMBRAL_RIESGO,
        "umbral_f1_max": comparacion["logistica"]["umbral_f1_max"]["umbral"],
        "poblacion_por_defecto": "activos",
        "importancia_permutacion_auc": importancia,
        "efectos_numericos": efectos_num,
        "efectos_categoricos": efectos_cat,
        "entrenado": datetime.now().isoformat(timespec="seconds"),
    }
    joblib.dump(artefacto, RUTA_MODELOS / "attrition.joblib")
    return {"elegido": "logistica", "descartado": "lightgbm", "comparacion": comparacion,
            "importancia_permutacion_auc": dict(sorted(importancia.items(), key=lambda kv: -kv[1])),
            "efectos_numericos": efectos_num, "efectos_categoricos": efectos_cat,
            "chequeo_activos": chequeo_activos}


# ==========================================================================
# 2. Ventas
# ==========================================================================

def modelo_lineal_ventas() -> TransformedTargetRegressor:
    lineal = Pipeline([("diseno", FunctionTransformer(diseno_ventas)),
                       ("lineal", LinearRegression(fit_intercept=False))])
    return TransformedTargetRegressor(regressor=lineal, func=np.log1p, inverse_func=np.expm1)


def modelo_interacciones_completas() -> TransformedTargetRegressor:
    # Alternativa: todas las interacciones de a pares (incluye región x clima y clima x cobre)
    pre = ColumnTransformer([("cat", OneHotEncoder(drop="first", sparse_output=False), ["categoria", "region"]),
                             ("num", StandardScaler(), VARIABLES_VENTAS)])
    reg = Pipeline([("pre", pre), ("int", PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)),
                    ("lineal", LinearRegression())])
    return TransformedTargetRegressor(regressor=reg, func=np.log1p, inverse_func=np.expm1)


def a_categorias(datos: pd.DataFrame) -> pd.DataFrame:
    datos = datos[COLUMNAS_VENTAS].copy()
    datos["categoria"] = pd.Categorical(datos["categoria"], categories=CATEGORIAS)
    datos["region"] = pd.Categorical(datos["region"], categories=REGIONES)
    return datos


def metricas_reg(real, pred) -> dict:
    real, pred = np.asarray(real, float), np.asarray(pred, float)
    return redondear({"mae": mean_absolute_error(real, pred), "rmse": mean_squared_error(real, pred) ** 0.5,
                      "mape_pct": np.mean(np.abs(real - pred) / np.maximum(real, 1)) * 100,
                      "r2": r2_score(real, pred)}, 3)


def entrenar_ventas() -> dict:
    ventas = pd.read_csv(RUTA_DATOS / "ventas.csv", parse_dates=["semana"])
    y = ventas["unidades_vendidas"]
    print(f"[ventas] {len(ventas)} filas, {ventas['semana'].nunique()} semanas "
          f"({ventas['semana'].min().date()} a {ventas['semana'].max().date()})")

    # Validación temporal: se entrena con las semanas viejas y se testea con el último 20 %
    semanas = np.sort(ventas["semana"].unique())
    corte = semanas[int(len(semanas) * 0.8)]
    tr, te = ventas["semana"] < corte, ventas["semana"] >= corte
    comparacion = {}

    lineal = modelo_lineal_ventas().fit(ventas.loc[tr, COLUMNAS_VENTAS], y[tr])
    pred_lineal = lineal.predict(ventas.loc[te, COLUMNAS_VENTAS])
    comparacion["lineal_log_interacciones_por_categoria"] = metricas_reg(y[te], pred_lineal)

    completas = modelo_interacciones_completas().fit(ventas.loc[tr, COLUMNAS_VENTAS], y[tr])
    comparacion["lineal_log_todas_las_interacciones"] = metricas_reg(y[te], completas.predict(ventas.loc[te, COLUMNAS_VENTAS]))

    gbm = lgb.LGBMRegressor(n_estimators=600, learning_rate=0.03, num_leaves=31, random_state=SEMILLA, verbose=-1)
    gbm.fit(a_categorias(ventas[tr]), y[tr])
    comparacion["lightgbm"] = metricas_reg(y[te], gbm.predict(a_categorias(ventas[te])))
    for nombre, m in comparacion.items():
        print(f"[ventas] {nombre:40s} MAE {m['mae']:6.1f} | MAPE {m['mape_pct']:5.1f}% | R2 {m['r2']:.3f}")

    test = ventas.loc[te, ["categoria", "unidades_vendidas"]].assign(pred=pred_lineal)
    test["error_pct"] = (test["unidades_vendidas"] - test["pred"]).abs() / test["unidades_vendidas"].clip(lower=1) * 100
    mape_por_categoria = test.groupby("categoria")["error_pct"].mean().round(1).to_dict()

    # Extrapolación: el cobre de los escenarios puede quedar fuera del rango histórico
    cobre_ultimo = float(ventas.loc[ventas["semana"] == semanas[-1], "precio_cobre_usd_ton"].iloc[0])
    rango_cobre = [float(ventas["precio_cobre_usd_ton"].min()), float(ventas["precio_cobre_usd_ton"].max())]

    # Modelo final: se entrena con las 200 semanas
    final = modelo_lineal_ventas().fit(ventas[COLUMNAS_VENTAS], y)
    coefs = dict(zip(nombres_diseno_ventas(), final.regressor_.named_steps["lineal"].coef_))
    efecto = {c: {v: float(coefs[f"{c} x {v}"]) for v in VARIABLES_VENTAS} for c in CATEGORIAS}
    residuo_log = np.log1p(y) - np.log1p(final.predict(ventas[COLUMNAS_VENTAS]))
    efectos_legibles = {c: redondear({
        "pct_por_1_grado": np.expm1(e["temperatura_media_c"]) * 100,
        "pct_por_10_mm_lluvia": np.expm1(10 * e["lluvia_mm"]) * 100,
        "pct_por_1000_usd_cobre": np.expm1(1000 * e["precio_cobre_usd_ton"]) * 100,
    }, 2) for c, e in efecto.items()}

    ultimas = ventas[ventas["semana"] >= semanas[-8]]
    artefacto = {
        "pipeline": final,
        "categorias": CATEGORIAS,
        "regiones": REGIONES,
        "variables": VARIABLES_VENTAS,
        "coef_log": efecto,                                   # efecto sobre log(unidades) por unidad de cada variable
        "efectos_legibles": efectos_legibles,
        "rangos": {v: [float(ventas[v].min()), float(ventas[v].max())] for v in VARIABLES_VENTAS},
        "rango_temperatura_por_region": {r: [float(g["temperatura_media_c"].min()), float(g["temperatura_media_c"].max())]
                                         for r, g in ventas.groupby("region")},
        "ultima_semana": str(pd.Timestamp(semanas[-1]).date()),
        "cobre_ultimo": cobre_ultimo,
        "clima_ultimas_8_semanas": {r: {"temperatura_media_c": round(float(g["temperatura_media_c"].mean()), 1),
                                        "lluvia_mm": round(float(g["lluvia_mm"].mean()), 1)}
                                    for r, g in ultimas.groupby("region")},
        "desvio_residuo_log": round(float(residuo_log.std()), 4),
        # Al volver de log a unidades la predicción queda en la mediana, un poco por debajo de la media.
        # Factor de Duan: se aplica solo a cantidades absolutas (los % de los escenarios no cambian).
        "factor_correccion_log": round(float(np.mean(np.exp(residuo_log))), 4),
        "entrenado": datetime.now().isoformat(timespec="seconds"),
    }
    joblib.dump(artefacto, RUTA_MODELOS / "ventas.joblib")
    print(f"[ventas] cobre último {cobre_ultimo:,.0f} USD/t; rango histórico {rango_cobre[0]:,.0f} a {rango_cobre[1]:,.0f}")
    return {"elegido": "lineal_log_interacciones_por_categoria", "descartados": ["lineal_log_todas_las_interacciones", "lightgbm"],
            "split_temporal": {"train_hasta": str(pd.Timestamp(corte).date()), "semanas_train": int(tr.sum() / 36),
                               "semanas_test": int(te.sum() / 36)},
            "comparacion_test": comparacion, "mape_pct_por_categoria_test": mape_por_categoria,
            "efectos_legibles": efectos_legibles, "cobre_ultimo": cobre_ultimo, "rango_cobre_historico": rango_cobre,
            "desvio_residuo_log": artefacto["desvio_residuo_log"],
            "factor_correccion_log": artefacto["factor_correccion_log"]}


if __name__ == "__main__":
    RUTA_MODELOS.mkdir(exist_ok=True)
    metricas = {"generado": datetime.now().isoformat(timespec="seconds"),
                "attrition": entrenar_attrition(), "ventas": entrenar_ventas()}
    with open(RUTA_MODELOS / "metricas.json", "w", encoding="utf-8") as f:
        json.dump(metricas, f, ensure_ascii=False, indent=2)
    print(f"Listo: modelos y métricas en {RUTA_MODELOS}")
